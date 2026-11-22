"""Draft-odds and hit models, validated leave-one-draft-class-out.

hit % = P(drafted) x P(top-N season within 5 years | drafted)
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.03, 0.1, 0.3, 1.0]
BASE = ['conf_strength', 'height', 'weight', 'est_draft_age', 'redshirt']
PROD = {
    'QB': ['any_a', 'cmp_pct', 'pass_att_pg', 'rush_yds_pg', 'int_rate', 'n_seasons'],
    'RB': ['scrim_pg', 'rec_share', 'ypc', 'rush_td_pg', 'rush_share', 'breakout_idx', 'best_share'],
    'WR': ['rec_yds_pg', 'best_share', 'yptpa', 'dominator', 'breakout_idx'],
    'TE': ['rec_yds_pg', 'best_share', 'yptpa', 'breakout_idx'],
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
ADV_DRAFT = {'QB': [], 'RB': [], 'WR': [], 'TE': ['rec_ypg_vs_power']}
ADV_HIT = {'QB': [], 'RB': ['rush_success', 'fd_per_carry'], 'WR': [], 'TE': ['rec_ypg_vs_power']}


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
        for stage in ['base', 'combine']:
            dcols, cols = feats(pos, stage, 'draft'), feats(pos, stage, 'hit')
            dm = _fit(dp, dp.drafted.values, dcols, dp.draft_year)
            hm = _fit(hp, hp.hit.values, cols, hp.draft_year)
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
                    tr = hp.draft_year != g
                    m = _model(hm['C']).fit(hp.loc[tr, cols], hp.hit.values[tr.values])
                    idx = miss & (ev.draft_year == g)
                    ev.loc[idx, 'p_hit_given_draft'] = m.predict_proba(ev.loc[idx, cols])[:, 1]
            ev['hit_pct'] = ev.p_draft * ev.p_hit_given_draft
            ev['outcome'] = ((ev.drafted == 1) & (ev.hit == 1)).astype(int)
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
                'overall_auc': round(roc_auc_score(ev.outcome, ev.hit_pct), 3),
                'overall_brier': round(brier_score_loss(ev.outcome, ev.hit_pct), 4),
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
    out['model_score'] = (out.groupby('pos').hit_pct.rank(pct=True) * 100).round(0)
    return out


def weights(fitted):
    rows = []
    for (pos, stage), f in fitted.items():
        for kind in ['draft', 'hit']:
            lr = f[kind]['model'].named_steps['logisticregression']
            for c, w in zip(f[kind]['cols'], lr.coef_[0]):
                rows.append({'pos': pos, 'stage': stage, 'model': kind, 'feature': c, 'weight': round(float(w), 3)})
    return pd.DataFrame(rows)
