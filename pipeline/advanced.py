"""Advanced stats built from play-level data (down, distance, field position, opponent, score, clock).

Per player-season:
  success rate            gain >= 50% of distance on 1st down, 70% on 2nd, 100% on 3rd/4th (or a TD)
  explosive rate          receptions of 20+ yards, rushes of 12+ yards, completions of 20+ yards
  first downs per opp     gains that reach the line to gain, per target / carry / dropback
  3rd-down share          share of the team's 3rd/4th-down targets (receivers) or carries (backs)
  red-zone share          share of the team's targets / carries inside the opponent's 20
  yards per target, catch rate
  vs strong defenses      production per game against Power-conference opponents
  competitive production  share of team yards outside garbage time (Q2 lead > 38, Q3 > 28, Q4 > 22)
"""
import numpy as np
import pandas as pd
from .config import DATA, STATS_SEASONS, POWER

COLS = ['game_id', 'team', 'conference', 'opponent', 'team_score', 'opponent_score', 'period', 'yards_to_goal', 'down',
        'distance', 'reception_player_id', 'reception_player', 'reception_yds', 'completion_player_id', 'completion_player',
        'completion_yds', 'rush_player_id', 'rush_yds', 'incompletion_player_id', 'target_player_id',
        'interception_thrown_player_id', 'sack_taken_player_id']


def _needed(down, dist):
    return np.where(down == 1, 0.5 * dist, np.where(down == 2, 0.7 * dist, dist))


def _season(yr):
    d = pd.read_csv(DATA / 'cfb' / f'player_stats_{yr}.csv', usecols=COLS, low_memory=False)
    # same passer/receiver swap fix as college.py
    as_passer = d.groupby('completion_player_id').size()
    as_recv = d.groupby('reception_player_id').size()
    qb_ids = set(as_passer.index[as_passer > as_recv.reindex(as_passer.index).fillna(0)])
    swap = d.reception_player_id.isin(qb_ids) & ~d.completion_player_id.isin(qb_ids) & d.completion_player_id.notna()
    for a, b in [('reception_player_id', 'completion_player_id'), ('reception_player', 'completion_player')]:
        d.loc[swap, [a, b]] = d.loc[swap, [b, a]].values

    conf = d.groupby('team').conference.first()
    d['opp_power'] = d.opponent.map(conf).isin(POWER).astype(int)
    diff = (d.team_score - d.opponent_score).abs()
    d['garbage'] = ((d.period == 2) & (diff > 38)) | ((d.period == 3) & (diff > 28)) | ((d.period >= 4) & (diff > 22))
    d['late_down'] = d.down.isin([3, 4])
    d['rz'] = d.yards_to_goal <= 20
    need = _needed(d.down, d.distance)
    has_dd = d.down.between(1, 4) & d.distance.notna()

    # ---- receiving targets ----
    rec = d[d.reception_player_id.notna()].assign(pid=lambda x: x.reception_player_id, yds=lambda x: x.reception_yds, caught=1)
    inc = d[d.target_player_id.notna() & d.reception_player_id.isna()].assign(pid=lambda x: x.target_player_id, yds=0.0, caught=0)
    t = pd.concat([rec, inc])
    tn = need.reindex(t.index).values if False else _needed(t.down, t.distance)
    t['succ'] = ((t.yds >= tn) | (t.yds >= t.yards_to_goal)) & t.down.between(1, 4)
    t['fd'] = (t.yds >= t.distance) & t.down.between(1, 4)
    t['expl'] = t.yds >= 20
    tg = t.groupby(['pid', 'team']).agg(
        tgts=('caught', 'size'), catches=('caught', 'sum'), t_yds=('yds', 'sum'), t_succ=('succ', 'sum'),
        t_fd=('fd', 'sum'), t_expl=('expl', 'sum'), t_late=('late_down', 'sum'), t_rz=('rz', 'sum'),
        rec_yds_vs_power=('yds', lambda s: s[t.loc[s.index, 'opp_power'] == 1].sum()),
        rec_yds_comp=('yds', lambda s: s[~t.loc[s.index, 'garbage']].sum()))
    team_t = t.groupby('team').agg(team_late_tgts=('late_down', 'sum'), team_rz_tgts=('rz', 'sum'),
                                   team_rec_yds_comp=('yds', lambda s: s[~t.loc[s.index, 'garbage']].sum()))

    # ---- rushing ----
    r = d[d.rush_player_id.notna()].copy()
    rn = _needed(r.down, r.distance)
    r['succ'] = ((r.rush_yds >= rn) | (r.rush_yds >= r.yards_to_goal)) & r.down.between(1, 4)
    r['fd'] = (r.rush_yds >= r.distance) & r.down.between(1, 4)
    r['expl'] = r.rush_yds >= 12
    rg = r.groupby(['rush_player_id', 'team']).agg(
        carries=('rush_yds', 'size'), r_succ=('succ', 'sum'), r_fd=('fd', 'sum'), r_expl=('expl', 'sum'),
        r_late=('late_down', 'sum'), r_rz=('rz', 'sum'),
        rush_yds_vs_power=('rush_yds', lambda s: s[r.loc[s.index, 'opp_power'] == 1].sum()),
        rush_yds_comp=('rush_yds', lambda s: s[~r.loc[s.index, 'garbage']].sum()))
    rg.index.names = ['pid', 'team']
    team_r = r.groupby('team').agg(team_late_carries=('late_down', 'sum'), team_rz_carries=('rz', 'sum'),
                                   team_rush_yds_comp=('rush_yds', lambda s: s[~r.loc[s.index, 'garbage']].sum()))

    # ---- QB dropbacks ----
    parts = [d[d.completion_player_id.notna()].assign(pid=lambda x: x.completion_player_id, yds=lambda x: x.completion_yds),
             d[d.incompletion_player_id.notna()].assign(pid=lambda x: x.incompletion_player_id, yds=0.0),
             d[d.interception_thrown_player_id.notna()].assign(pid=lambda x: x.interception_thrown_player_id, yds=0.0),
             d[d.sack_taken_player_id.notna()].assign(pid=lambda x: x.sack_taken_player_id, yds=-1.0)]
    q = pd.concat(parts)
    qn = _needed(q.down, q.distance)
    q['succ'] = ((q.yds >= qn) | (q.yds >= q.yards_to_goal)) & q.down.between(1, 4) & (q.yds > 0)
    q['fd'] = (q.yds >= q.distance) & q.down.between(1, 4) & (q.yds > 0)
    q['expl'] = q.yds >= 20
    qg = q.groupby(['pid', 'team']).agg(dropbacks=('yds', 'size'), q_succ=('succ', 'sum'), q_fd=('fd', 'sum'),
                                        q_expl=('expl', 'sum'),
                                        q_late_conv=('fd', lambda s: s[q.loc[s.index, 'late_down']].sum()),
                                        q_late=('late_down', 'sum'))

    games_vs_power = d[d.opp_power == 1].groupby('team').game_id.nunique().rename('team_games_vs_power')
    out = pd.concat([tg, rg, qg], axis=1).reset_index().rename(columns={'level_0': 'pid'})
    out = out.merge(team_t, left_on='team', right_index=True, how='left') \
             .merge(team_r, left_on='team', right_index=True, how='left') \
             .merge(games_vs_power, left_on='team', right_index=True, how='left')
    out['season'] = yr
    out['player_id'] = out.pid.astype('int64')
    return out.drop(columns='pid')


def build_advanced():
    raw = pd.concat([_season(y) for y in STATS_SEASONS], ignore_index=True).fillna(0)
    a = raw
    z = lambda num, den: np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    f = pd.DataFrame({'player_id': a.player_id, 'season': a.season, 'team': a.team})
    # receiving
    f['tgt_success'] = z(a.t_succ, a.tgts)
    f['rec_explosive'] = z(a.t_expl, a.tgts)
    f['fd_per_tgt'] = z(a.t_fd, a.tgts)
    f['yds_per_tgt'] = z(a.t_yds, a.tgts)
    f['catch_rate'] = z(a.catches, a.tgts)
    f['late_tgt_share'] = z(a.t_late, a.team_late_tgts)
    f['rz_tgt_share'] = z(a.t_rz, a.team_rz_tgts)
    f['rec_comp_share'] = z(a.rec_yds_comp, a.team_rec_yds_comp)
    f['rec_ypg_vs_power'] = z(a.rec_yds_vs_power, a.team_games_vs_power)
    # rushing
    f['rush_success'] = z(a.r_succ, a.carries)
    f['rush_explosive'] = z(a.r_expl, a.carries)
    f['fd_per_carry'] = z(a.r_fd, a.carries)
    f['late_carry_share'] = z(a.r_late, a.team_late_carries)
    f['rz_carry_share'] = z(a.r_rz, a.team_rz_carries)
    f['rush_comp_share'] = z(a.rush_yds_comp, a.team_rush_yds_comp)
    f['scrim_ypg_vs_power'] = z(a.rush_yds_vs_power + a.rec_yds_vs_power, a.team_games_vs_power)
    # QB
    f['qb_success'] = z(a.q_succ, a.dropbacks)
    f['qb_explosive'] = z(a.q_expl, a.dropbacks)
    f['qb_fd_per_db'] = z(a.q_fd, a.dropbacks)
    f['qb_late_conv'] = z(a.q_late_conv, a.q_late)
    f['qb_rush_ypg_vs_power'] = z(a.rush_yds_vs_power, a.team_games_vs_power)
    # minimum opportunity filters
    f.loc[a.tgts < 15, ['tgt_success', 'rec_explosive', 'fd_per_tgt', 'yds_per_tgt', 'catch_rate']] = np.nan
    f.loc[a.carries < 20, ['rush_success', 'rush_explosive', 'fd_per_carry']] = np.nan
    f.loc[a.dropbacks < 50, ['qb_success', 'qb_explosive', 'qb_fd_per_db', 'qb_late_conv']] = np.nan
    f.loc[a.team_games_vs_power < 2, ['rec_ypg_vs_power', 'scrim_ypg_vs_power', 'qb_rush_ypg_vs_power']] = np.nan
    return f


ADV_COLS = ['tgt_success', 'rec_explosive', 'fd_per_tgt', 'yds_per_tgt', 'catch_rate', 'late_tgt_share', 'rz_tgt_share',
            'rec_comp_share', 'rec_ypg_vs_power', 'rush_success', 'rush_explosive', 'fd_per_carry', 'late_carry_share',
            'rz_carry_share', 'rush_comp_share', 'scrim_ypg_vs_power', 'qb_success', 'qb_explosive', 'qb_fd_per_db',
            'qb_late_conv', 'qb_rush_ypg_vs_power']


def attach(df, adv):
    """Join advanced stats for each row's final season on player id + school."""
    m = df.merge(adv, left_on=['player_id', 'final_season', 'college_team'], right_on=['player_id', 'season', 'team'],
                 how='left', suffixes=('', '_adv'))
    return m.drop(columns=[c for c in ['season', 'team'] if c in m.columns and c not in df.columns])
