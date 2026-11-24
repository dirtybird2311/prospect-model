"""Recruiting rankings (247Sports Composite via CollegeFootballData.com; files pulled by scripts/fetch_recruiting.py).

Matched to college players by CFBD athlete id (the same id the college stats use), else by name + committed
school + class year within a year of his first college roster season.
"""
import glob
import numpy as np
import pandas as pd
from .config import DATA
from .features import norm, school_tokens


def load_recruits():
    files = sorted(glob.glob(str(DATA / 'recruiting' / 'recruits_*.csv')))
    if not files:
        return None
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return r[r.rating.notna()]


def recruit_table(cs):
    """player_id -> recruit_rating, recruit_stars (best rating if he appears more than once)."""
    r = load_recruits()
    if r is None:
        return None
    byid = r[r.athleteId.notna()].assign(player_id=lambda x: x.athleteId.astype('int64'))[['player_id', 'rating', 'stars']]
    # fallback: name + committed school + class year
    first = cs.groupby('player_id').agg(name=('name', 'first'), first_seen=('season', 'min'))
    teams = cs.groupby('player_id').team.apply(lambda s: set().union(*[school_tokens(t) for t in s.unique()]))
    first['key'], first['tok'] = first.name.map(norm), teams
    first = first[~first.index.isin(byid.player_id)].reset_index()
    rn = r[r.athleteId.isna()].assign(key=lambda x: x.name.map(norm), tok=lambda x: x.committedTo.fillna('').map(school_tokens))
    m = first.merge(rn[['key', 'tok', 'year', 'rating', 'stars']], on='key', suffixes=('', '_r'))
    m = m[(m.year - m.first_seen).abs() <= 1]
    m = m[[len(a & b) > 0 for a, b in zip(m.tok, m.tok_r)]]
    m = m.groupby('player_id').filter(lambda g: len(g) == 1)[['player_id', 'rating', 'stars']]
    t = pd.concat([byid, m]).sort_values('rating', ascending=False).drop_duplicates('player_id')
    return t.rename(columns={'rating': 'recruit_rating', 'stars': 'recruit_stars'}).set_index('player_id')


def attach_recruiting(df, rt):
    out = df.copy()
    if rt is None:
        out['recruit_rating'] = np.nan
        out['recruit_stars'] = np.nan
    else:
        pid = out.player_id.astype('int64')
        out['recruit_rating'] = pid.map(rt.recruit_rating)
        out['recruit_stars'] = pid.map(rt.recruit_stars)
    # unrated recruits (walk-ons, JUCO, unranked) get the bottom of the scale plus a flag
    out['unrated'] = out.recruit_rating.isna().astype(int)
    out['recruit_rating'] = out.recruit_rating.fillna(0.70)
    out['recruit_stars'] = out.recruit_stars.fillna(1)
    return out
