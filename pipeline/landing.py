"""Landing spot for drafted players: what his NFL team had at his position, and how good its offense was.

  vac_share   share of the team's previous-season opportunity at his position (QB pass attempts, RB carries +
              targets, WR/TE targets) from players not on its roster in his rookie season (nflverse season roster,
              which lists each player's last team that year)
  team_off    team's previous-season half-PPR points from QBs, RBs, WRs and TEs, per game
  true_age    age on Sept. 1 of his rookie season, from NFL birth dates (known once he's drafted)
"""
import numpy as np
import pandas as pd
from .config import DATA, POS

FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA',
       # draft_picks uses Pro Football Reference codes
       'GNB': 'GB', 'KAN': 'KC', 'NOR': 'NO', 'SFO': 'SF', 'NWE': 'NE', 'TAM': 'TB', 'SDG': 'LAC', 'LVR': 'LV',
       'RAI': 'LV', 'RAM': 'LA', 'PHO': 'ARI'}


def _team(s):
    return s.replace(FIX)


def _opp(st):
    f = lambda c: st[c].fillna(0) if c in st else 0
    return np.select([st.position == 'QB', st.position == 'RB'], [f('attempts'), f('carries') + f('targets')], f('targets'))


def landing_table():
    """(team, season, position) -> vac_share, team_off for each season with a previous season of stats."""
    rows = []
    for y in range(2015, 2027):
        sp = DATA / 'nfl' / f'stats_{y - 1}.csv'
        rp = DATA / 'nfl' / 'rosters' / f'roster_{y}.csv'
        if not sp.exists() or not rp.exists():
            continue
        st = pd.read_csv(sp, low_memory=False)
        st = st[st.position.isin(POS)].copy()
        st['team'] = _team(st.recent_team if 'recent_team' in st else st.team)
        st['opp'] = _opp(st)
        from .lookback import half_ppr
        st['pts'] = half_ppr(st)
        ro = pd.read_csv(rp, low_memory=False, usecols=['season', 'team', 'gsis_id', 'week'])
        ro['team'] = _team(ro.team)
        on = set(zip(ro.team, ro.gsis_id))
        st['back'] = [(t, g) in on for t, g in zip(st.team, st.player_id)]
        g = st.groupby(['team', 'position'])
        t = pd.DataFrame({'opp': g.opp.sum(), 'ret': st[st.back].groupby(['team', 'position']).opp.sum()}).fillna(0)
        t['vac_share'] = 1 - t.ret / t.opp.replace(0, np.nan)
        games = st.groupby('team').games.max()
        t = t.reset_index()
        t['team_off'] = t.team.map(st.groupby('team').pts.sum() / games)
        t['season'] = y
        rows.append(t[['team', 'season', 'position', 'vac_share', 'team_off']])
    return pd.concat(rows, ignore_index=True)


def attach_landing(df):
    """Add vac_share, team_off and true_age to drafted rows (needs draft_team, draft_year, gsis_id)."""
    lt = landing_table()
    out = df.copy()
    out['_t'] = _team(out.draft_team.fillna(''))
    out = out.merge(lt, left_on=['_t', 'draft_year', 'pos'], right_on=['team', 'season', 'position'], how='left',
                    suffixes=('', '_l')).drop(columns=['_t', 'team', 'season', 'position'], errors='ignore')
    pl = pd.read_csv(DATA / 'nfl' / 'players.csv', low_memory=False, usecols=['gsis_id', 'birth_date'])
    bd = pd.to_datetime(pl.drop_duplicates('gsis_id').set_index('gsis_id').birth_date, errors='coerce')
    born = pd.to_datetime(out.gsis_id.map(bd))
    out['true_age'] = (pd.to_datetime(out.draft_year.astype('Int64').astype(str) + '-09-01', errors='coerce') - born).dt.days / 365.25
    return out
