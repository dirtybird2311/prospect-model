"""Roll college play-level stats up to player-season totals with team context, plus roster info."""
import numpy as np
import pandas as pd
from .config import DATA, STATS_SEASONS, ROSTER_SEASONS

COLS = ['game_id', 'team', 'conference',
        'reception_player_id', 'reception_player', 'reception_yds',
        'completion_player_id', 'completion_player', 'completion_yds',
        'rush_player_id', 'rush_player', 'rush_yds',
        'interception_thrown_player_id', 'touchdown_player_id',
        'incompletion_player_id', 'target_player_id']


def _season(yr):
    d = pd.read_csv(DATA / 'cfb' / f'player_stats_{yr}.csv', usecols=COLS, low_memory=False)
    # Source quirk: on some pass plays (mostly TDs) passer and receiver are swapped.
    as_passer = d.groupby('completion_player_id').size()
    as_recv = d.groupby('reception_player_id').size()
    qb_ids = set(as_passer.index[as_passer > as_recv.reindex(as_passer.index).fillna(0)])
    swap = d.reception_player_id.isin(qb_ids) & ~d.completion_player_id.isin(qb_ids) & d.completion_player_id.notna()
    for a, b in [('reception_player_id', 'completion_player_id'), ('reception_player', 'completion_player')]:
        d.loc[swap, [a, b]] = d.loc[swap, [b, a]].values
    td = d.touchdown_player_id
    pass_td = td.notna() & d.completion_player_id.notna() & ((td == d.reception_player_id) | (td == d.completion_player_id))
    d['rec_td'] = pass_td.astype(int)
    d['pass_td'] = d['rec_td']
    d['rush_td'] = (td.notna() & (td == d.rush_player_id) & d.completion_player_id.isna()).astype(int)

    team = d.groupby('team').agg(
        t_games=('game_id', 'nunique'), t_rec=('reception_player_id', 'count'), t_rec_yds=('reception_yds', 'sum'),
        t_rush_att=('rush_player_id', 'count'), t_rush_yds=('rush_yds', 'sum'), t_cmp=('completion_player_id', 'count'),
        t_inc=('incompletion_player_id', 'count'), t_int=('interception_thrown_player_id', 'count'),
        t_rec_td=('rec_td', 'sum'), t_rush_td=('rush_td', 'sum'), conference=('conference', 'first'))
    team['t_pass_att'] = team.t_cmp + team.t_inc + team.t_int

    def agg(idcol, namecol, **spec):
        sub = d[d[idcol].notna()]
        g = sub.groupby([idcol, 'team']).agg(**spec)
        g['name'] = sub.groupby([idcol, 'team'])[namecol].agg(lambda s: s.mode().iat[0])
        g.index.names = ['player_id', 'team']
        return g

    def count(idcol, name):
        s = d[d[idcol].notna()].groupby([idcol, 'team']).size().rename(name)
        s.index.names = ['player_id', 'team']
        return s

    rec = agg('reception_player_id', 'reception_player', rec=('reception_yds', 'count'), rec_yds=('reception_yds', 'sum'),
              rec_td=('rec_td', 'sum'), g1=('game_id', 'nunique'))
    rush = agg('rush_player_id', 'rush_player', rush_att=('rush_yds', 'count'), rush_yds=('rush_yds', 'sum'),
               rush_td=('rush_td', 'sum'), g2=('game_id', 'nunique'))
    cmp_ = agg('completion_player_id', 'completion_player', cmp=('completion_yds', 'count'), pass_yds=('completion_yds', 'sum'),
               pass_td=('pass_td', 'sum'), g3=('game_id', 'nunique'))
    p = pd.concat([rec, rush.drop(columns='name'), cmp_.drop(columns='name'), count('target_player_id', 'inc_tgt'),
                   count('incompletion_player_id', 'inc'), count('interception_thrown_player_id', 'ints')], axis=1)
    p['name'] = pd.concat([rec['name'], rush['name'], cmp_['name']]).groupby(level=[0, 1]).first()
    p = p.fillna({c: 0 for c in p.columns if c != 'name'})
    p['tgt'] = p.rec + p.inc_tgt
    p['pass_att'] = p.cmp + p.inc + p.ints
    p['games'] = p[['g1', 'g2', 'g3']].max(axis=1)
    p = p.drop(columns=['g1', 'g2', 'g3', 'inc_tgt', 'inc']).reset_index()
    p = p.merge(team, left_on='team', right_index=True, how='left')
    p['season'] = yr
    p['player_id'] = p.player_id.astype('int64')
    return p


def load_rosters():
    out = []
    for yr in ROSTER_SEASONS:
        f = DATA / 'cfb' / f'cfb_rosters_{yr}.csv'
        if not f.exists():
            continue
        r = pd.read_csv(f, low_memory=False, usecols=lambda c: c in {'athlete_id', 'team', 'weight', 'height', 'year', 'position'})
        r['season'] = yr
        out.append(r)
    r = pd.concat(out).rename(columns={'athlete_id': 'player_id', 'position': 'col_pos', 'year': 'class_yr'})
    r['player_id'] = pd.to_numeric(r.player_id, errors='coerce')
    r = r.dropna(subset=['player_id'])
    r['player_id'] = r.player_id.astype('int64')
    r['class_yr'] = pd.to_numeric(r.class_yr, errors='coerce').where(lambda s: s.between(1, 5))
    return r.drop_duplicates(['player_id', 'season'])


def build_college():
    """Player-season table (main team row per season) with roster fields merged in."""
    cs = pd.concat([_season(yr) for yr in STATS_SEASONS], ignore_index=True)
    ros = load_rosters()
    cs = cs.merge(ros[['player_id', 'season', 'weight', 'height', 'class_yr', 'col_pos']], on=['player_id', 'season'], how='left')
    cs = cs.sort_values('games', ascending=False).drop_duplicates(['player_id', 'season'])
    first_seen = pd.concat([ros[['player_id', 'season']], cs[['player_id', 'season']]]).groupby('player_id').season.min()
    cs['first_seen'] = cs.player_id.map(first_seen)
    last_pos = ros.dropna(subset=['col_pos']).sort_values('season').groupby('player_id').col_pos.last()
    cs['listed_pos'] = cs.player_id.map(last_pos)
    return cs, ros
