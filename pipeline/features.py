"""Turn college player-seasons into model rows: the draft pool, NFL outcomes and the current prospect pool."""
import re
import numpy as np
import pandas as pd
from .config import DATA, POS, POWER, FBS, MIN_USAGE, HIT_WINDOW, HIT_CUT, PROSPECT_SEASON, NFL_STAT_SEASONS

ALIASES = {'central florida': 'ucf', 'ala-birmingham': 'uab', 'texas-el paso': 'utep', 'texas-san antonio': 'utsa',
           'southern methodist': 'smu', 'texas christian': 'tcu', 'brigham young': 'byu', 'louisiana state': 'lsu',
           'southern california': 'usc', 'north carolina st.': 'nc state', 'miami (fl)': 'miami', 'miami (oh)': 'miami oh',
           'mississippi': 'ole miss', 'pittsburgh': 'pitt pittsburgh', 'bowling green': 'bowling green state',
           'boston col.': 'boston college', 'middle tenn. st.': 'middle tennessee', 'fla. atlantic': 'florida atlantic',
           'fla. international': 'florida international fiu', 'western ky.': 'western kentucky',
           'southern miss': 'southern mississippi', 'la-monroe': 'louisiana monroe', 'la-lafayette': 'louisiana',
           'nevada-las vegas': 'unlv', 'hawaii': 'hawaii hawai'}


def norm(name):
    s = re.sub(r"[.'\-]", '', str(name).lower())
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def school_tokens(s):
    s = str(s).lower().strip()
    s = ALIASES.get(s, s).replace('st.', 'state').replace('&', '')
    return set(re.findall(r'[a-z]+', s)) - {'university', 'of', 'the', 'college'}


def infer_pos(r):
    if r.listed_pos in POS:
        return r.listed_pos
    if r.pass_att >= 50:
        return 'QB'
    return 'RB' if r.rush_att > 2 * r.rec else 'WR'


def qualifies(r, pos):
    col, n = MIN_USAGE[pos]
    return r[col] >= n


# ---------------- features ----------------
def season_feats(r):
    g = max(r.games, 1)
    pa = r.pass_att if r.pass_att >= 50 else np.nan
    f = {
        'games': r.games, 'power': int(r.conference in POWER), 'height': r.height, 'weight': r.weight,
        'rec_pg': r.rec / g, 'rec_yds_pg': r.rec_yds / g, 'rec_td_pg': r.rec_td / g,
        'yds_share': r.rec_yds / r.t_rec_yds if r.t_rec_yds > 0 else 0,
        'td_share': r.rec_td / r.t_rec_td if r.t_rec_td > 0 else 0,
        'tgt_share': r.tgt / r.t_pass_att if r.t_pass_att > 0 else 0,
        'yptpa': r.rec_yds / r.t_pass_att if r.t_pass_att > 0 else 0,
        'rec_share': r.rec / r.t_rec if r.t_rec > 0 else 0,
        'rush_yds_pg': r.rush_yds / g, 'rush_td_pg': r.rush_td / g,
        'ypc': r.rush_yds / r.rush_att if r.rush_att >= 20 else np.nan,
        'rush_share': r.rush_yds / r.t_rush_yds if r.t_rush_yds > 0 else 0,
        'scrim_pg': (r.rush_yds + r.rec_yds) / g, 'pass_att_pg': r.pass_att / g,
        'any_a': (r.pass_yds + 20 * r.pass_td - 45 * r.ints) / pa,
        'cmp_pct': r.cmp / pa, 'td_rate': r.pass_td / pa, 'int_rate': r.ints / pa,
        'class_yr': r.class_yr,
    }
    f['dominator'] = (f['yds_share'] + f['td_share']) / 2
    return f


def career_feats(hist, pos):
    h = hist[hist.games >= 4]
    if pos in ('WR', 'TE'):
        share, thresh = (h.rec_yds / h.t_rec_yds.replace(0, np.nan)).fillna(0), (0.20 if pos == 'WR' else 0.12)
    elif pos == 'RB':
        share, thresh = ((h.rush_yds + h.rec_yds) / (h.t_rush_yds + h.t_rec_yds).replace(0, np.nan)).fillna(0), 0.20
    else:
        share, thresh = (h.pass_att / h.t_pass_att.replace(0, np.nan)).fillna(0), 0.60
    hit = np.where(share.values >= thresh)[0]
    return {'best_share': share.max() if len(share) else 0,
            'breakout_idx': hit[0] if len(hit) else len(share) + 1,
            'n_seasons': len(h)}


def profile(cs_by_pid, pid, final_season, pos, conf_rate):
    hist = cs_by_pid.get_group(pid)
    hist = hist[hist.season <= final_season].sort_values('season')
    fin = hist[hist.season == final_season].iloc[0]
    out = season_feats(fin)
    out.update(career_feats(hist, pos))
    yrs = final_season - fin.first_seen + 1
    out['yrs_enrolled'] = yrs
    out['est_draft_age'] = 18.4 + (final_season + 1 - fin.first_seen)  # enroll ~18.3, draft in April after final season
    out['redshirt'] = int(pd.notna(fin.class_yr) and yrs - fin.class_yr >= 1)
    out['conf_strength'] = conf_rate.get(fin.conference, np.nan)
    out.update({'player_id': pid, 'college_team': fin.team, 'conference': fin.conference, 'final_season': final_season,
                'name': fin['name'], 'pos': pos})
    return out


# ---------------- matching draft picks ----------------
def match_draft(cs):
    dp = pd.read_csv(DATA / 'nfl' / 'draft_picks.csv')
    dp = dp[dp.season.between(2015, PROSPECT_SEASON) & dp.position.isin(POS)].copy()
    dp['key'] = dp.pfr_player_name.map(norm)
    cs = cs.assign(key=cs.name.map(norm), last=cs.name.map(lambda n: norm(n).split()[-1] if norm(n) else ''),
                   tok=cs.team.map(school_tokens))
    rows = []
    for _, p in dp.iterrows():
        ctok = school_tokens(p.college)
        win = cs.season.between(p.season - 2, p.season - 1)
        cand = cs[win & (cs.key == p.key)]
        if ctok:
            cand = cand[cand.tok.map(lambda t: len(t & ctok) > 0)]
        if cand.empty:
            last = p.key.split()[-1] if p.key else ''
            cand = cs[win & (cs['last'] == last) & (cs.games >= 4) &
                      cs.tok.map(lambda t: len(t & ctok) > 0 and len(t & ctok) >= len(ctok) - 1)]
            if cand.player_id.nunique() > 1 and p.key:
                # several same-surname players at the school (Mike vs Michael Washington): break the tie on first initial
                cand = cand[cand.key.str[:1] == p.key[:1]]
            if cand.player_id.nunique() != 1:
                continue
        pid = cand.assign(o=cand.tok.map(lambda t: len(t & ctok))).sort_values(['o', 'season', 'games'], ascending=False).player_id.iloc[0]
        rows.append({'player_id': pid, 'draft_year': p.season, 'pick': p.pick, 'gsis_id': p.gsis_id,
                     'nfl_pos': p.position, 'seasons_started': p.seasons_started, 'draft_name': p.pfr_player_name,
                     'team': p.team})
    m = pd.DataFrame(rows).drop_duplicates('player_id')
    print(f'matched {len(m)} of {len(dp)} drafted QB/RB/WR/TE to college stats')
    return m


# ---------------- combine ----------------
def combine_table(cs_lookup):
    """Combine rows matched to college player ids by name + school + draft year."""
    cb = pd.read_csv(DATA / 'nfl' / 'combine.csv')
    cb = cb[cb.pos.isin(POS) & cb.season.between(2015, PROSPECT_SEASON + 1)].copy()
    cb['key'] = cb.player_name.map(norm)

    def inches(h):
        try:
            a, b = str(h).split('-')
            return int(a) * 12 + int(b)
        except Exception:
            return np.nan

    cb['c_ht'] = cb.ht.map(inches)
    cb['speed_score'] = cb.wt * 200 / cb.forty ** 4
    out = []
    for _, r in cb.iterrows():
        cand = cs_lookup[(cs_lookup.key == r.key) & cs_lookup.season.between(r.season - 2, r.season - 1)]
        tok = school_tokens(r.school)
        if tok:
            c2 = cand[cand.tok.map(lambda t: len(t & tok) > 0)]
            cand = c2 if len(c2) else cand
        if cand.player_id.nunique() == 1:
            out.append({'player_id': cand.player_id.iloc[0], 'combine_year': r.season, 'forty': r.forty,
                        'vertical': r.vertical, 'broad_jump': r.broad_jump, 'cone': r.cone,
                        'speed_score': r.speed_score, 'c_wt': r.wt})
    return pd.DataFrame(out).drop_duplicates('player_id')


# ---------------- pools ----------------
def build_pools(cs):
    cs = cs[cs.games >= 1].copy()
    cs['pos'] = cs.apply(infer_pos, axis=1)
    by = cs.groupby('player_id')
    last_season = by.season.max()

    drafted = match_draft(cs)
    # conference strength: drafted skill players per team-season, by conference (2014-2025 seasons)
    team_seasons = cs[cs.season <= 2025].groupby('conference').apply(lambda g: g[['team', 'season']].drop_duplicates().shape[0])
    d_conf = cs.merge(drafted[['player_id', 'draft_year']], on='player_id')
    d_conf = d_conf[d_conf.season == d_conf.draft_year - 1].groupby('conference').size()
    conf_rate = (d_conf / team_seasons).fillna(0).to_dict()

    lookup = cs.assign(key=cs.name.map(norm), tok=cs.team.map(school_tokens))[['player_id', 'season', 'key', 'tok']]
    cb = combine_table(lookup)

    # ---- draft pool: everyone whose college career ended 2014-2025, plus drafted players ----
    rows = []
    dmap = drafted.set_index('player_id')
    for pid, fin_season in last_season.items():
        is_drafted = pid in dmap.index
        if fin_season > 2025 and not is_drafted:
            continue
        hist = by.get_group(pid)
        if is_drafted:
            dy = dmap.loc[pid, 'draft_year']
            prior = hist[hist.season <= dy - 1]
            if prior.empty:
                continue
            f = prior[prior.season == dy - 1]
            fin = f.iloc[0] if (len(f) and f.games.iloc[0] >= 4) else prior.sort_values('games').iloc[-1]
        else:
            f = hist[hist.season == fin_season].iloc[0]
            prev = hist[hist.season == fin_season - 1]
            fin = f
            if not qualifies(f, f.pos) and len(prev) and f.games < 5 and qualifies(prev.iloc[0], prev.iloc[0].pos):
                fin = prev.iloc[0]
        pos = dmap.loc[pid, 'nfl_pos'] if is_drafted else fin.pos
        if fin.conference not in FBS or pos not in POS:
            continue
        if not is_drafted and not qualifies(fin, pos):
            continue
        r = profile(by, pid, int(fin.season), pos, conf_rate)
        r['draft_year'] = int(dmap.loc[pid, 'draft_year']) if is_drafted else int(fin_season) + 1
        r['drafted'] = int(is_drafted)
        r['pick'] = dmap.loc[pid, 'pick'] if is_drafted else np.nan
        r['gsis_id'] = dmap.loc[pid, 'gsis_id'] if is_drafted else None
        r['draft_team'] = dmap.loc[pid, 'team'] if (is_drafted and 'team' in dmap.columns) else None
        r['seasons_started'] = dmap.loc[pid, 'seasons_started'] if is_drafted else np.nan
        rows.append(r)
    pool = pd.DataFrame(rows)
    pool = pool.merge(cb, on='player_id', how='left')
    pool.loc[pool.combine_year != pool.draft_year, ['forty', 'vertical', 'broad_jump', 'cone', 'speed_score', 'c_wt']] = np.nan
    pool['at_combine'] = (pool.combine_year == pool.draft_year).astype(int)

    # ---- undrafted players who signed with an NFL team (any roster, practice squad included) ----
    # nflverse's player table carries ESPN ids, the same ids the college stats use.
    pl = pd.read_csv(DATA / 'nfl' / 'players.csv', low_memory=False, usecols=['gsis_id', 'espn_id', 'rookie_season', 'draft_pick', 'display_name', 'college_name'])
    pl = pl[pl.draft_pick.isna() & pl.gsis_id.notna()].copy()
    plid = pl[pd.to_numeric(pl.espn_id, errors='coerce').notna()].copy()
    plid['espn_id'] = pd.to_numeric(plid.espn_id).astype('int64')
    umap = plid.drop_duplicates('espn_id').set_index('espn_id')
    pid = pool.player_id.astype('int64')
    yr = pid.map(umap.rookie_season)
    signed = (pool.drafted == 0) & pid.isin(umap.index) & (yr.isna() | yr.between(pool.draft_year - 1, pool.draft_year + 1))
    pool['udfa_signed'] = signed.astype(int)
    pool.loc[signed, 'gsis_id'] = pid[signed].map(umap.gsis_id)
    # older college ids aren't ESPN ids: fall back to name + college + rookie season within a year
    pl['key'] = pl.display_name.map(norm)
    pl['tok'] = pl.college_name.fillna('').map(lambda c: set().union(*[school_tokens(x.strip()) for x in c.split(';')]) if c else set())
    rest = pool[(pool.drafted == 0) & (pool.udfa_signed == 0)].assign(key=lambda x: x.name.map(norm))
    mm = rest[['player_id', 'key', 'college_team', 'draft_year']].merge(pl[['key', 'tok', 'gsis_id', 'rookie_season']], on='key')
    mm = mm[(mm.rookie_season - mm.draft_year).abs() <= 1]
    mm = mm[[len(school_tokens(t) & k) > 0 for t, k in zip(mm.college_team, mm.tok)]]
    mm = mm[~mm.gsis_id.isin(pool.gsis_id.dropna())].groupby('player_id').filter(lambda g: len(g) == 1).set_index('player_id')
    fb = pool.player_id.isin(mm.index)
    pool.loc[fb, 'udfa_signed'] = 1
    pool.loc[fb, 'gsis_id'] = pool.loc[fb, 'player_id'].map(mm.gsis_id)

    # ---- outcomes for drafted players and signed undrafted players ----
    st = pd.concat([pd.read_csv(DATA / 'nfl' / f'stats_{y}.csv', low_memory=False) for y in NFL_STAT_SEASONS])
    st['pts'] = (st.passing_yards.fillna(0) * 0.04 + st.passing_tds.fillna(0) * 4 - st.passing_interceptions.fillna(0) * 2
                 + st.rushing_yards.fillna(0) * 0.1 + st.rushing_tds.fillna(0) * 6 + st.receptions.fillna(0) * 0.5
                 + st.receiving_yards.fillna(0) * 0.1 + st.receiving_tds.fillna(0) * 6
                 - (st.rushing_fumbles_lost.fillna(0) + st.receiving_fumbles_lost.fillna(0) + st.sack_fumbles_lost.fillna(0)) * 2)
    st = st[st.position.isin(POS)]
    st['rank'] = st.groupby(['season', 'position']).pts.rank(ascending=False, method='first')
    st['startable'] = st['rank'] <= st.position.map(HIT_CUT)
    last_nfl = max(NFL_STAT_SEASONS)
    # points at the hit line (RB24, WR24, QB12, TE12) each season, so a season can be scored relative to it
    cutpts = st[st['rank'] == st.position.map(HIT_CUT)].set_index(['season', 'position']).pts
    st['rel'] = st.pts / st.set_index(['season', 'position']).index.map(cutpts).values
    hits = []
    for _, r in pool[(pool.drafted == 1) | (pool.udfa_signed == 1)].iterrows():
        s = st[(st.player_id == r.gsis_id) & st.season.between(r.draft_year, r.draft_year + HIT_WINDOW - 1)]
        first = s[s.startable].season.min() if s.startable.any() else np.nan
        hits.append((r.player_id, int(s.startable.any()), s['rank'].min() if len(s) else np.nan,
                     s.rel.max() if len(s) else 0.0, first - r.draft_year + 1 if first == first else np.nan))
    h = pd.DataFrame(hits, columns=['player_id', 'hit', 'best_rank', 'best_rel', 'yrs_to_hit'])
    pool = pool.merge(h, on='player_id', how='left')
    pool['hit_complete'] = pool.draft_year + HIT_WINDOW - 1 <= last_nfl   # full window observed
    pool['seasons_obs'] = (last_nfl - pool.draft_year + 1).clip(upper=HIT_WINDOW)   # NFL seasons seen so far

    # ---- current prospects ----
    eligible = cs[(cs.season == PROSPECT_SEASON)].player_id.unique()
    prow = []
    for pid in eligible:
        hist = by.get_group(pid)
        r26 = hist[hist.season == PROSPECT_SEASON].iloc[0]
        if r26.first_seen > PROSPECT_SEASON - 2 or r26.conference not in FBS or pid in dmap.index:
            continue
        r25 = hist[hist.season == PROSPECT_SEASON - 1]
        pos = r26.pos
        if pos not in POS:
            continue
        if r26.games >= 4 and qualifies(r26, pos):
            use = r26
        elif len(r25) and qualifies(r25.iloc[0], pos) and r26.games < 7 and r25.iloc[0].conference in FBS:
            use = r25.iloc[0]
        else:
            continue
        p = profile(by, pid, int(use.season), pos, conf_rate)
        p['name'] = r26['name']
        p['college_team'] = r26.team
        p['draft_year'] = PROSPECT_SEASON + 1
        p['used_prior_season'] = int(use.season != PROSPECT_SEASON)
        prow.append(p)
    pros = pd.DataFrame(prow).merge(cb, on='player_id', how='left')
    pros.loc[pros.combine_year != pros.draft_year, ['forty', 'vertical', 'broad_jump', 'cone', 'speed_score', 'c_wt']] = np.nan
    pros['at_combine'] = (pros.combine_year == pros.draft_year).astype(int)
    # transfer: played for more than one school through his final season (WR/TE transfers hit less, Nov 2026 test)
    for df in (pool, pros):
        m = df[['player_id', 'final_season']].merge(cs[['player_id', 'season', 'team']], on='player_id')
        n = m[m.season <= m.final_season].groupby('player_id').team.nunique()
        df['transfer'] = (df.player_id.map(n) > 1).astype(int)
    return pool, pros, conf_rate
