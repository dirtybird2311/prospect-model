"""Lookback: how the model would have rated the most recent draft class, against what's happened since.

Draft and hit models are refit without that class (the hit model never sees it anyway: it trains on
classes with a full 5-year window), and players are scored on college stats only, as of the end of
their final college season, before the combine. Their NFL results come from nflverse's current-season
player stats, ranked by half-PPR points at their position.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .config import DATA, POS, HIT_CUT
from . import models as M


def half_ppr(st):
    f = lambda c: st[c].fillna(0) if c in st else 0
    return (f('passing_yards') * 0.04 + f('passing_tds') * 4 - f('passing_interceptions') * 2
            + f('rushing_yards') * 0.1 + f('rushing_tds') * 6 + f('receptions') * 0.5
            + f('receiving_yards') * 0.1 + f('receiving_tds') * 6
            - (f('rushing_fumbles_lost') + f('receiving_fumbles_lost') + f('sack_fumbles_lost')) * 2)


def lookback(pool, season):
    """Score the class drafted in `season` with models that never saw it; attach its `season` NFL results."""
    path = DATA / 'nfl' / f'stats_{season}.csv'
    if not path.exists() or not (pool.draft_year == season).any():
        return None, None
    fitted, _, _ = M.fit_all(pool[pool.draft_year < season])
    cls = pool[pool.draft_year == season].copy()
    cls['at_combine'] = 0
    sc = M.score(fitted, cls)

    st = pd.read_csv(path, low_memory=False)
    st = st[st.position.isin(POS)].copy()
    st['pts'] = half_ppr(st)
    st['nfl_rank'] = st.groupby('position').pts.rank(ascending=False, method='min')
    st['ppg'] = st.pts / st.games.clip(lower=1)
    sc = sc.merge(st[['player_id', 'pts', 'ppg', 'games', 'nfl_rank']]
                  .rename(columns={'player_id': 'gsis_id', 'games': 'nfl_games'}), on='gsis_id', how='left')
    sc['model_rank'] = sc.groupby('pos').hit_pct.rank(ascending=False, method='first').astype(int)
    sc['pick_rank'] = sc[sc.drafted == 1].groupby('pos').pick.rank(method='first')
    cut = sc.pos.map(HIT_CUT)
    sc['status'] = np.select(
        [sc.drafted != 1, sc.nfl_rank.isna(), sc.nfl_rank <= cut, sc.nfl_rank <= cut * 2.5],
        ['Not drafted', 'No stats yet', 'On hit pace', 'Contributing'], 'Not producing')

    summ = []
    d = sc[sc.drafted == 1].assign(pts=lambda x: x.pts.fillna(0))
    for pos in ['QB', 'RB', 'WR', 'TE', 'All']:
        x = d if pos == 'All' else d[d.pos == pos]
        if len(x) < 4:
            continue
        top = x[x.model_rank <= 10]
        rest = x[x.model_rank > 10]
        summ.append({'id': pos, 'pos': pos, 'drafted': len(x),
                     'model_corr': round(float(spearmanr(x.hit_pct, x.pts)[0]), 2),
                     'pick_corr': round(float(spearmanr(-x.pick, x.pts)[0]), 2),
                     'top10_pts': round(float(top.pts.mean()), 1) if len(top) else None,
                     'rest_pts': round(float(rest.pts.mean()), 1) if len(rest) else None,
                     'on_pace': int((x.status == 'On hit pace').sum()),
                     'weeks': int(st.games.max())})
    return sc, pd.DataFrame(summ)
