"""Write dashboard data files: JSON {rows, meta}, each carrying the code that produced it."""
import json
import numpy as np
import pandas as pd
from .config import OUT, ROOT, PROSPECT_SEASON

LABELS = {'any_a': 'Adjusted yards per attempt', 'cmp_pct': 'Completion %', 'pass_att_pg': 'Pass attempts per game',
          'rush_yds_pg': 'Rushing yards per game', 'int_rate': 'Interception rate', 'n_seasons': 'College seasons played',
          'conf_strength': 'Conference strength', 'height': 'Height', 'weight': 'Weight',
          'est_draft_age': 'Age at draft (est.)', 'redshirt': 'Redshirted', 'scrim_pg': 'Scrimmage yards per game',
          'rec_share': 'Share of team catches', 'ypc': 'Yards per carry', 'rush_td_pg': 'Rushing TDs per game',
          'rush_share': 'Share of team rushing yards', 'breakout_idx': 'Late breakout (seasons before)',
          'best_share': 'Best season share of team production', 'rec_yds_pg': 'Receiving yards per game',
          'yptpa': 'Yards per team pass attempt', 'dominator': 'Share of team rec. yards + TDs',
          'at_combine': 'Combine invite', 'speed_score': 'Speed score', 'vertical': 'Vertical jump',
          'broad_jump': 'Broad jump', 'forty': '40-yard dash',
          'rec_yds_pg_x_conf': 'Receiving yards/game x conference strength', 'best_share_x_conf': 'Best season share x conference strength',
          'any_a_x_conf': 'Adj. yards/att x conference strength', 'cmp_pct_x_conf': 'Completion % x conference strength',
          'rush_yds_pg_x_conf': 'Rushing yards/game x conference strength', 'scrim_pg_x_conf': 'Scrimmage yards/game x conference strength',
          'rec_share_x_conf': 'Share of team catches x conference strength',
          'rush_success': 'Rushing success rate', 'fd_per_carry': 'First downs per carry',
          'rec_ypg_vs_power': 'Receiving yards/game vs Power defenses'}


def _code():
    files = ['config.py', 'fetch.py', 'college.py', 'features.py', 'advanced.py', 'models.py', 'export.py']
    return [{'label': f, 'value': (ROOT / 'pipeline' / f).read_text(), 'type': 'code', 'lang': 'python'} for f in files]


def _write(name, df, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(df.replace([np.inf, -np.inf], np.nan).to_json(orient='records'))
    (OUT / f'{name}.json').write_text(json.dumps({'rows': rows, 'meta': meta}))
    print('wrote', name, len(rows))


def key_stats(r):
    if r.pos == 'QB':
        return f"{r.any_a:.1f} adj. yds/att, {r.cmp_pct*100:.0f}% comp, {r.rush_yds_pg:.0f} rush yds/g"
    if r.pos == 'RB':
        ypc = '' if pd.isna(r.ypc) else f", {r.ypc:.1f} yds/carry"
        succ = '' if pd.isna(r.get('rush_success')) else f", {r.rush_success*100:.0f}% rush success"
        return f"{r.scrim_pg:.0f} scrimmage yds/g, {r.rec_share*100:.0f}% of team catches{ypc}{succ}"
    vp = '' if pd.isna(r.get('rec_ypg_vs_power')) else f", {r.rec_ypg_vs_power:.0f} yds/g vs Power defenses"
    return f"{r.rec_yds_pg:.0f} rec yds/g, {r.yds_share*100:.0f}% of team rec yds, best {r.best_share*100:.0f}%{vp}"


def export(scored, val, back, w, run_at):
    src = [{'label': 'College stats', 'value': f'sportsdataverse/cfbfastR-data player_stats + rosters, 2014-{PROSPECT_SEASON} (current season through the latest weekly update). Some bowl and playoff games are missing.'},
           {'label': 'NFL data', 'value': 'nflverse draft_picks, combine, stats_player_reg'},
           {'label': 'Run at', 'value': run_at}]
    code = _code()

    s = scored.copy()
    s['key_stats'] = s.apply(key_stats, axis=1)
    s['rank'] = s.groupby('pos').hit_pct.rank(ascending=False, method='first').astype(int)
    for c in ['p_draft', 'p_hit_given_draft', 'hit_pct']:
        s[c] = (s[c] * 100).round(1)
    s['player_id'] = s.player_id.astype(str)
    keep = ['player_id', 'rank', 'name', 'pos', 'college_team', 'conference', 'final_season', 'used_prior_season',
            'est_draft_age', 'rush_success', 'fd_per_carry', 'rec_ypg_vs_power', 'tgt_success', 'rec_explosive', 'yds_per_tgt', 'p_draft', 'p_hit_given_draft', 'hit_pct', 'model_score', 'stage', 'key_stats', 'games']
    _write('prospects', s[keep].sort_values(['pos', 'rank']), src + [
        {'label': 'Definitions', 'value': 'Hit % = draft odds x hit odds if drafted. A hit is a top-24 RB/WR or top-12 QB/TE half-PPR season within his first 5 NFL seasons. Model score = percentile of hit % among current prospects at his position. Draft odds treat this season as his last in college.'}] + code)

    v = val.copy()
    v['stage_label'] = v.stage.map({'base': 'College stats', 'combine': 'Stats + combine'})
    v['id'] = v.pos + '-' + v.stage
    for c in ['drafted_top_q_hit', 'drafted_bottom_q_hit', 'base_rate', 'cal_top_decile_pred', 'cal_top_decile_actual']:
        v[c] = (v[c] * 100).round(1)
    _write('validation', v, src + [
        {'label': 'Method', 'value': 'Leave-one-draft-class-out: each class is scored by models fit on the others. Draft model: every FBS QB/RB/WR/TE with a real role whose college career ended 2014-2025. Hit model: drafted players from 2015-2021 (full 5-year window). AUC = chance a random hit is ranked above a random miss (0.5 = coin flip). Quarter hit rates are among drafted players, ranked by hit %.'}] + code)

    w = w[(w.stage == 'base')].copy()
    w['feature_label'] = w.feature.map(LABELS).fillna(w.feature)
    w['id'] = w.pos + '-' + w.model + '-' + w.feature
    _write('weights', w, src + [{'label': 'Units', 'value': 'Standardized logistic-regression weights (log-odds per standard deviation). Positive helps. Correlated inputs share credit.'}] + code)

    b = back.copy()
    b['id'] = b.player_id.astype(str)
    b['hit_pct'] = (b.hit_pct * 100).round(1)
    b['score_pct'] = (b.groupby('pos').hit_pct.rank(pct=True) * 100).round(0)
    b = b[b.drafted == 1][['id', 'name', 'pos', 'college_team', 'draft_year', 'pick', 'hit', 'best_rank', 'hit_pct', 'score_pct']]
    _write('backtest', b.sort_values(['draft_year', 'pick']), src + [
        {'label': 'Method', 'value': "Drafted players 2015-2021. Each player's hit % comes from models that never saw his draft class. score_pct ranks him among all pool players at his position, before the draft."}] + code)
