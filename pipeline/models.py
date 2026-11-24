"""Draft-odds and hit models, validated leave-one-draft-class-out.

hit % = P(drafted) x P(top-N season within 5 years | drafted)
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.03, 0.1, 0.3, 1.0]
BASE = ['conf_strength', 'height', 'weight', 'est_draft_age', 'redshirt']
PROD = {
    'QB': ['any_a', 'cmp_pct', 'pass_att_pg', 'rush_yds_pg', 'int_rate', 'n_seasons'],
    'RB': ['scrim_pg', 'rec_share', 'ypc', 'rush_td_pg', 'rush_share', 'breakout_idx', 'best_share'],
    'WR': ['rec_yds_pg', 'best_share', 'yptpa', 'dominator', 'breakout_idx', 'transfer'],
    'TE': ['rec_yds_pg', 'best_share', 'yptpa', 'breakout_idx', 'transfer'],
}
# Conference adjustment: production x conference strength, so the same numbers count for more against
# stronger competition. Backtest showed G5 WR/TE/QB production carried over poorly; G5 pass-catching RBs did fine.
CONF_X = {
    'QB': ['any_a_x_conf', 'cmp_pct_x_conf', 'rush_yds_pg_x_conf'],
    'RB': ['scrim_pg_x_conf', 'rec_share_x_conf'],
    'WR': ['rec_yds_pg_x_conf', 'best_share_x_conf'],
    'TE': ['rec_yds_pg_x_conf', 'best_share_x_conf'],
}
USE_CONF_X = {'QB': False, 'RB': False, 'WR': True, 'TE': True}   # kept where it improved the backtest (Nov 21, 2026)


def add_derived(df):
    df = df.copy()
    cs = df['conf_strength']
    for base in ['any_a', 'cmp_pct', 'rush_yds_pg', 'scrim_pg', 'rec_share', 'rec_yds_pg', 'best_share']:
        df[f'{base}_x_conf'] = df[base] * cs
    return df


COMBINE = {
    'QB': ['at_combine'],
    'RB': ['at_combine', 'speed_score', 'vertical', 'broad_jump'],
    'WR': ['at_combine', 'forty'],
    'TE': ['at_combine', 'speed_score', 'vertical', 'broad_jump'],
}


# Advanced stats (pipeline/advanced.py), added separately to the draft and hit models where the backtest improved.
# Kept Nov 21, 2026 after testing all seven advanced stats one at a time and in combination.
# Recruiting (247 Composite rating, pipeline/recruiting.py), added Nov 23, 2026: helped RB draft and hit models
# (hit AUC .758 -> .775) and the WR hit model with an unrated flag (.703 -> .710); not QB or TE, and nothing
# post-draft (draft slot already prices it in).
ADV_DRAFT = {'QB': [], 'RB': ['recruit_rating'], 'WR': [], 'TE': ['rec_ypg_vs_power']}
ADV_HIT = {'QB': [], 'RB': ['rush_success', 'fd_per_carry', 'recruit_rating'], 'WR': ['recruit_rating', 'unrated'],
           'TE': ['rec_ypg_vs_power']}


def feats(pos, stage, kind='hit'):
    f = PROD[pos] + BASE + (CONF_X[pos] if USE_CONF_X.get(pos) else [])
    f = f + (ADV_DRAFT if kind == 'draft' else ADV_HIT)[pos]
    return f + COMBINE[pos] if stage == 'combine' else f


def _model(C):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def _oof(d, y, cols, C, groups):
    p = np.full(len(d), np.nan)
    for g in sorted(groups.unique()):
        tr, te = (groups != g).values, (groups == g).values
        if y[tr].sum() < 3:
            p[te] = y[tr].mean()
            continue
        p[te] = _model(C).fit(d.loc[tr, cols], y[tr]).predict_proba(d.loc[te, cols])[:, 1]
    return p


# Hit model target. 'hit' = logistic regression on hit/miss. 'points' = ridge regression on log best-season
# points relative to the hit line (RB24/WR24/QB12/TE12), calibrated to hit odds, and also trained on recent
# classes that haven't had their full 5 years (their best season so far, MIN_PARTIAL_SEASONS+ NFL seasons).
# Tested Nov 23, 2026 (leave-one-class-out on 2015-2021): hit AUC QB .749 -> .789, WR .627 -> .700,
# TE .610 -> .670; RB .758 -> .746, so RB keeps hit/miss.
HIT_TARGET = {'QB': 'points', 'RB': 'hit', 'WR': 'points', 'TE': 'points'}
ALPHA_GRID = [1, 10, 30, 100, 300]
MIN_PARTIAL_SEASONS = 2


def _ridge(a):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=a))


def _points_y(d):
    return np.log(0.1 + d.best_rel.fillna(0).values)


class PointsHit:
    """Ridge on log best-season points vs the hit line, mapped to hit odds by a logistic calibrator."""

    def __init__(self, reg, cal):
        self.reg, self.cal = reg, cal

    def predict_proba(self, X):
        p = self.cal.predict_proba(self.reg.predict(X).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])

    @property
    def named_steps(self):
        return self.reg.named_steps


def _points_raw_oof(d, extra, cols, a):
    """Leave-one-class-out ridge predictions for d's classes (extra rows always train)."""
    raw = np.full(len(d), np.nan)
    for g in sorted(d.draft_year.unique()):
        t = pd.concat([d[d.draft_year != g], extra], ignore_index=True)
        te = (d.draft_year == g).values
        raw[te] = _ridge(a).fit(t[cols], _points_y(t)).predict(d.loc[te, cols])
    return raw


def _points_fit(d, extra, cols, a):
    tr = pd.concat([d, extra], ignore_index=True)
    reg = _ridge(a).fit(tr[cols], _points_y(tr))
    raw = _points_raw_oof(d, extra, cols, a)
    cal = LogisticRegression(max_iter=5000).fit(raw.reshape(-1, 1), d.hit.values)
    return PointsHit(reg, cal)


def _hit_model(d, extra, cols, C, kind):
    if kind == 'points':
        return _points_fit(d, extra, cols, C)
    return _model(C).fit(d[cols], d.hit.values)


def _hit_oof(d, extra, cols, C, kind):
    if kind == 'points':
        # one calibrator over all out-of-fold predictions: keeps each class's ranking as the ridge made it
        raw = _points_raw_oof(d, extra, cols, C)
        return LogisticRegression(max_iter=5000).fit(raw.reshape(-1, 1), d.hit.values).predict_proba(raw.reshape(-1, 1))[:, 1]
    p = np.full(len(d), np.nan)
    for g in sorted(d.draft_year.unique()):
        tr = d[d.draft_year != g].reset_index(drop=True)
        te = (d.draft_year == g).values
        if kind != 'points' and tr.hit.sum() < 3:
            p[te] = tr.hit.mean()
            continue
        p[te] = _hit_model(tr, extra, cols, C, kind).predict_proba(d.loc[te, cols])[:, 1]
    return p


def _fit_hit(d, extra, cols, pos):
    kind = HIT_TARGET[pos]
    extra = extra if kind == 'points' else extra.iloc[:0]
    best = None
    for C in (ALPHA_GRID if kind == 'points' else C_GRID):
        o = _hit_oof(d, extra, cols, C, kind)
        a = roc_auc_score(d.hit.values, o)
        if best is None or a > best[0]:
            best = (a, C, o)
    auc, C, oof = best
    return {'model': _hit_model(d, extra, cols, C, kind), 'cols': cols, 'C': C, 'auc': auc, 'oof': oof,
            'kind': kind, 'extra': extra}


def _fit(d, y, cols, groups):
    best = None
    for C in C_GRID:
        o = _oof(d, y, cols, C, groups)
        a = roc_auc_score(y, o)
        if best is None or a > best[0]:
            best = (a, C, o)
    auc, C, oof = best
    return {'model': _model(C).fit(d[cols], y), 'cols': cols, 'C': C, 'auc': auc, 'oof': oof}


def fit_all(pool):
    """Fit draft and hit models for each position and stage. Returns (fitted, validation_df, backtest_df)."""
    fitted, val, back = {}, [], []
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if not (pool.pos == pos).any():
            continue
        dp = add_derived(pool[pool.pos == pos]).reset_index(drop=True)
        hp = dp[(dp.drafted == 1) & dp.hit_complete].reset_index(drop=True)
        hx = dp[(dp.drafted == 1) & ~dp.hit_complete & (dp.seasons_obs >= MIN_PARTIAL_SEASONS)].reset_index(drop=True)
        for stage in ['base', 'combine']:
            dcols, cols = feats(pos, stage, 'draft'), feats(pos, stage, 'hit')
            dm = _fit(dp, dp.drafted.values, dcols, dp.draft_year)
            hm = _fit_hit(hp, hx, cols, pos)
            fitted[(pos, stage)] = {'draft': dm, 'hit': hm}
            # combined out-of-sample check on classes with a full 5-year window
            ev = dp.assign(p_draft=dm['oof'])
            ev = ev[ev.draft_year <= hp.draft_year.max()].copy()
            hit_oof = pd.Series(hm['oof'], index=hp.player_id)
            # hit odds for undrafted players: score with a model that never saw their class
            ev['p_hit_given_draft'] = ev.player_id.map(hit_oof)
            miss = ev.p_hit_given_draft.isna()
            if miss.any():
                for g in ev.loc[miss, 'draft_year'].unique():
                    m = _hit_model(hp[hp.draft_year != g].reset_index(drop=True), hm['extra'], cols, hm['C'], hm['kind'])
                    idx = miss & (ev.draft_year == g)
                    ev.loc[idx, 'p_hit_given_draft'] = m.predict_proba(ev.loc[idx, cols])[:, 1]
            ev['hit_pct'] = ev.p_draft * ev.p_hit_given_draft
            ev['outcome'] = ((ev.drafted == 1) & (ev.hit == 1)).astype(int)
            # recalibration: map raw hit % to observed hit rates (Platt scaling on the log-odds), checked by class
            raw = ev.hit_pct.values.copy()
            brier_raw = brier_score_loss(ev.outcome, raw)
            cal_oof = np.full(len(ev), np.nan)
            for g in sorted(ev.draft_year.unique()):
                tr, te = (ev.draft_year != g).values, (ev.draft_year == g).values
                c = LogisticRegression(C=1e6, max_iter=5000).fit(_logit(raw[tr]).reshape(-1, 1), ev.outcome.values[tr])
                cal_oof[te] = c.predict_proba(_logit(raw[te]).reshape(-1, 1))[:, 1]
            brier_cal = brier_score_loss(ev.outcome, cal_oof)
            use_cal = brier_cal < 0.98 * brier_raw   # only where it clearly helps (2%+ better Brier score)
            fitted[(pos, stage)]['cal'] = (LogisticRegression(C=1e6, max_iter=5000)
                                           .fit(_logit(raw).reshape(-1, 1), ev.outcome.values) if use_cal else None)
            if use_cal:
                ev['hit_pct'] = cal_oof
            drafted_ev = ev[ev.drafted == 1].copy()
            drafted_ev['q'] = pd.qcut(drafted_ev.hit_pct.rank(method='first'), 4, labels=[1, 2, 3, 4])
            qrate = drafted_ev.groupby('q', observed=True).hit.mean()
            # calibration on full pool, by predicted decile
            ev['bin'] = pd.qcut(ev.hit_pct.rank(method='first'), 10, labels=False)
            cal = ev.groupby('bin').agg(pred=('hit_pct', 'mean'), actual=('outcome', 'mean'))
            val.append({
                'pos': pos, 'stage': stage,
                'draft_auc': round(dm['auc'], 3), 'draft_C': dm['C'], 'draft_n': len(dp), 'drafted': int(dp.drafted.sum()),
                'hit_auc': round(hm['auc'], 3), 'hit_C': hm['C'], 'hit_n': len(hp), 'hits': int(hp.hit.sum()),
                'hit_target': hm['kind'], 'partial_rows': len(hm['extra']),
                'overall_auc': round(roc_auc_score(ev.outcome, ev.hit_pct), 3),
                'overall_brier': round(brier_score_loss(ev.outcome, ev.hit_pct), 4),
                'brier_raw': round(brier_raw, 4), 'brier_recalibrated': round(brier_cal, 4), 'recalibrated': bool(use_cal),
                'base_rate': round(ev.outcome.mean(), 4),
                'drafted_top_q_hit': round(qrate.loc[4], 3), 'drafted_bottom_q_hit': round(qrate.loc[1], 3),
                'cal_top_decile_pred': round(cal.pred.iloc[-1], 3), 'cal_top_decile_actual': round(cal.actual.iloc[-1], 3),
            })
            if stage == 'base':
                back.append(ev[['player_id', 'name', 'pos', 'college_team', 'draft_year', 'drafted', 'pick', 'hit',
                                'best_rank', 'p_draft', 'p_hit_given_draft', 'hit_pct']])
    return fitted, pd.DataFrame(val), pd.concat(back, ignore_index=True)


def score(fitted, pros):
    """Score current prospects. Uses the combine stage where a prospect has combine results."""
    out = add_derived(pros)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        for stage in ['base', 'combine']:
            m = (out.pos == pos) & ((out.at_combine == 1) if stage == 'combine' else (out.at_combine != 1))
            if not m.any():
                continue
            f = fitted[(pos, stage)]
            out.loc[m, 'p_draft'] = f['draft']['model'].predict_proba(out.loc[m, f['draft']['cols']])[:, 1]
            out.loc[m, 'p_hit_given_draft'] = f['hit']['model'].predict_proba(out.loc[m, f['hit']['cols']])[:, 1]
            out.loc[m, 'stage'] = stage
    out['hit_pct'] = out.p_draft * out.p_hit_given_draft
    for (pos, stage), f in fitted.items():
        m = (out.pos == pos) & (out.stage == stage) & out.hit_pct.notna()
        if f.get('cal') is not None and m.any():
            out.loc[m, 'hit_pct'] = f['cal'].predict_proba(_logit(out.loc[m, 'hit_pct'].values).reshape(-1, 1))[:, 1]
    out['model_score'] = (out.groupby('pos').hit_pct.rank(pct=True) * 100).round(0)
    return out


# ---------------- post-draft stage ----------------
# Once a player is drafted, hit odds = f(draft slot, college hit score). Tested Nov 22, 2026 (nested leave-one-class-out):
# the college score added to draft slot for QB (AUC .893 -> .902) and RB (.834 -> .849) but not WR (.831 -> .801)
# or TE (.860 -> .856), so WR/TE post-draft odds use draft slot alone. Explosive-run rate, drafted RB teammates and
# committee-back flags were also tested and left out.
POST_COLLEGE = {'QB': True, 'RB': True, 'WR': False, 'TE': False}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _post_X(log_pick, college_p, pos):
    return np.column_stack([log_pick, _logit(college_p)]) if POST_COLLEGE[pos] else np.column_stack([log_pick])


def _stack():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def fit_post(pool, fitted):
    """Post-draft hit models per position. Returns (post_models, validation_df)."""
    post, val = {}, []
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if (pos, 'base') not in fitted:
            continue
        dp = add_derived(pool[pool.pos == pos]).reset_index(drop=True)
        d = dp[(dp.drafted == 1) & dp.hit_complete].reset_index(drop=True)
        hm = fitted[(pos, 'base')]['hit']
        cols, C = hm['cols'], hm['C']
        y, yrs, lp = d.hit.values, d.draft_year, np.log(d.pick.values)
        oof, oof_pick = np.full(len(d), np.nan), np.full(len(d), np.nan)
        for g in sorted(yrs.unique()):
            tr = (yrs != g).values
            te = ~tr
            dt = d[tr].reset_index(drop=True)
            inner = _hit_oof(dt, hm['extra'], cols, C, hm['kind'])   # college scores the stacker trains on, out of fold
            sc_te = _hit_model(dt, hm['extra'], cols, C, hm['kind']).predict_proba(d.loc[te, cols])[:, 1]
            oof[te] = _stack().fit(_post_X(lp[tr], inner, pos), y[tr]).predict_proba(_post_X(lp[te], sc_te, pos))[:, 1]
            oof_pick[te] = _stack().fit(lp[tr, None], y[tr]).predict_proba(lp[te, None])[:, 1]
        post[pos] = _stack().fit(_post_X(lp, hm['oof'], pos), y)
        q = pd.qcut(pd.Series(oof).rank(method='first'), 4, labels=False).values
        val.append({'pos': pos, 'stage': 'post', 'hit_auc': round(roc_auc_score(y, oof), 3),
                    'pick_only_auc': round(roc_auc_score(y, oof_pick), 3), 'hit_n': len(d), 'hits': int(y.sum()),
                    'uses_college_score': POST_COLLEGE[pos],
                    'drafted_top_q_hit': round(y[q == 3].mean(), 3), 'drafted_bottom_q_hit': round(y[q == 0].mean(), 3)})
    return post, pd.DataFrame(val)


def score_post(post, fitted, df):
    """Post-draft hit odds for rows with a draft pick."""
    out = add_derived(df)
    out['post_hit_pct'] = np.nan
    for pos, st in post.items():
        m = (out.pos == pos) & out.pick.notna()
        if not m.any():
            continue
        hm = fitted[(pos, 'base')]['hit']
        college = hm['model'].predict_proba(out.loc[m, hm['cols']])[:, 1]
        out.loc[m, 'post_hit_pct'] = st.predict_proba(_post_X(np.log(out.loc[m, 'pick'].values), college, pos))[:, 1]
    return out


def weights(fitted):
    rows = []
    for (pos, stage), f in fitted.items():
        for kind in ['draft', 'hit']:
            steps = f[kind]['model'].named_steps
            est = steps['logisticregression'] if 'logisticregression' in steps else steps['ridge']
            for c, w in zip(f[kind]['cols'], np.ravel(est.coef_)):
                rows.append({'pos': pos, 'stage': stage, 'model': kind, 'feature': c, 'weight': round(float(w), 3),
                             'target': 'drafted' if kind == 'draft' else f[kind].get('kind', 'hit')})
    return pd.DataFrame(rows)
