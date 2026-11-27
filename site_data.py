"""Export the data the Statable site (Astro, separate repo) is built from.

    python3 site_data.py [SITE_REPO]      # default ../statable

Reads data/scored.pkl, data/fitted.pkl, data/pool_adv.pkl and out/*.json (written by run.py) and writes:
    SITE_REPO/src/data/site.json      as-of date, top 5 per position, weekly movers
    SITE_REPO/src/data/players.json   one record per player page (odds, tiers, drivers, comps, odds history)
    SITE_REPO/src/data/track.json     validation, backtest best calls/misses, 2026 lookback
    SITE_REPO/public/data/board.json  the full board, loaded by the board page in the browser
    SITE_REPO/public/data/sleeper_map.json  2026 rookies keyed by Sleeper player id, for league sync
Only Statable's own odds go public: no raw 247Sports ratings, no college stat tables. Each run also saves a dated
snapshot of every prospect's odds under data/history/ for movers and odds over time.
"""
import glob
import io
import json
import math
import pickle
import re
import sys
import unicodedata
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import models as M
from pipeline.config import DATA, OUT
from pipeline.export import LABELS

ROOT = Path(__file__).parent
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'statable'
TODAY = date.today()
ASOF = f"{TODAY.strftime('%b')} {TODAY.day}, {TODAY.year}"
CLASS = 2027
TIERS = {'RB': ['RB1', 'RB2', 'Flex'], 'WR': ['WR1', 'WR2', 'WR3', 'Flex']}
HIDE_VALUE = {'recruit_rating', 'unrated', 'at_combine'}          # named as factors, values never shown
IDS_URL = 'https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv'


def slugify(*parts):
    s = unicodedata.normalize('NFKD', ' '.join(parts)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', s).strip('-')


def r1(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 1)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(',', ':'), ensure_ascii=False))


# ---------------------------------------------------------------- model outputs
def load():
    scored = pd.read_pickle(DATA / 'scored.pkl').copy()
    fitted, post = pickle.load(open(DATA / 'fitted.pkl', 'rb'))
    pool = pd.read_pickle(DATA / 'pool_adv.pkl')
    pj = {r['player_id']: r for r in json.load(open(OUT / 'prospects.json'))['rows']}
    scored['pid'] = scored.player_id.astype(str)
    for c in ['hit_pct', 'p_draft', 'p_hit_given_draft', 'tier1', 'tier2', 'tier3', 'tier4']:
        scored[c + '_100'] = scored[c] * 100 if c in scored else np.nan
    scored['rank'] = scored.pid.map(lambda i: pj.get(i, {}).get('rank'))
    scored['key_stats'] = scored.pid.map(lambda i: pj.get(i, {}).get('key_stats', ''))
    scored['slug'] = [slugify(n, c) for n, c in zip(scored.name, scored.college_team)]
    dup = scored.slug.duplicated(keep=False)
    scored.loc[dup, 'slug'] = scored.loc[dup, 'slug'] + '-' + scored.loc[dup, 'pid']
    return scored, fitted, pool


def tiers_of(r):
    if r.pos not in TIERS:
        return None
    return [r1(getattr(r, f'tier{k + 1}_100')) for k in range(len(TIERS[r.pos]))]


def model_parts(m):
    steps = m.named_steps
    est = steps['logisticregression'] if 'logisticregression' in steps else steps['ridge']
    return steps['simpleimputer'], steps['standardscaler'], np.ravel(est.coef_)


def drivers(fitted, scored):
    """Each input's pull on a prospect's odds (share of each model's total pull, draft + hit summed)."""
    out = {}
    df = M.add_derived(scored)
    for (pos, stage), f in fitted.items():
        m = (df.pos == pos) & (df.stage == stage)
        if not m.any():
            continue
        total = None
        for kind in ['draft', 'hit']:
            imp, sc, coef = model_parts(f[kind]['model'])
            cols = f[kind]['cols']
            c = sc.transform(imp.transform(df.loc[m, cols])) * coef
            c = c / np.maximum(np.abs(c).sum(axis=1, keepdims=True), 1e-9)
            part = pd.DataFrame(c, columns=cols, index=df.index[m])
            total = part if total is None else total.add(part, fill_value=0)
        for idx, row in total.iterrows():
            out[df.at[idx, 'pid']] = (row.sort_values(), df.loc[idx])
    return out


def fmt_value(col, v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or col in HIDE_VALUE:
        return ''
    if col.endswith('_x_conf') or col in ('conf_strength', 'breakout_idx'):
        return ''
    if col in ('rec_share', 'rush_share', 'best_share', 'dominator', 'cmp_pct', 'rush_success', 'int_rate'):
        return f'{v * 100:.0f}%'
    if col == 'est_draft_age':
        return f'{v:.1f} at the draft'
    if col in ('redshirt', 'transfer'):
        return 'yes' if v else 'no'
    if col == 'height':
        return f'{int(v) // 12}\'{int(v) % 12}"'
    if col == 'weight':
        return f'{v:.0f} lb'
    if abs(v) >= 10:
        return f'{v:.0f}'
    return f'{v:.2f}' if abs(v) < 1 else f'{v:.1f}'


def driver_lists(series, row):
    def pick(items):
        return [{'label': LABELS.get(k, k), 'value': fmt_value(k, row.get(k))} for k, _ in items]
    up = [(k, v) for k, v in series[::-1][:3].items() if v > 0.02]
    down = [(k, v) for k, v in series[:3].items() if v < -0.02]
    return {'up': pick(up), 'down': pick(down)}


def best_finish(b, pos):
    if b != b:
        return 'no NFL stats'
    b = int(b)
    cuts = {'RB': [(12, 'RB1'), (24, 'RB2'), (36, 'flex')], 'WR': [(12, 'WR1'), (24, 'WR2'), (36, 'WR3'), (47, 'flex')]}
    for cut, lab in cuts.get(pos, [(12, f'{pos}1')]):
        if b <= cut:
            return f'{pos}{b} best season ({lab})'
    return f'{pos}{b} best season'


def comps(fitted, pool, scored):
    """Three most similar drafted prospects from 2015-2021, by distance in the draft and hit models' weighted inputs."""
    out = {}
    hist = M.add_derived(pool[(pool.drafted == 1) & pool.hit_complete])
    cur = M.add_derived(scored)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        h, c = hist[hist.pos == pos], cur[cur.pos == pos]
        if not len(h) or not len(c):
            continue
        zh, zc = [], []
        for kind in ['draft', 'hit']:                    # draft-model inputs keep a top-10 pick from matching day-3 picks
            f = fitted[(pos, 'base')][kind]
            imp, sc, coef = model_parts(f['model'])
            w = np.abs(coef) / np.abs(coef).sum()
            zh.append(sc.transform(imp.transform(h[f['cols']])) * w)
            zc.append(sc.transform(imp.transform(c[f['cols']])) * w)
        zh, zc = np.hstack(zh), np.hstack(zc)
        d = ((zc[:, None, :] - zh[None, :, :]) ** 2).sum(axis=2)
        for i, pid in enumerate(c.pid):
            out[pid] = [{'name': h.iloc[j]['name'], 'college': h.iloc[j]['college_team'], 'year': int(h.iloc[j]['draft_year']),
                         'pick': int(h.iloc[j]['pick']), 'hit': bool(h.iloc[j]['hit'] == 1),
                         'finish': best_finish(h.iloc[j]['best_rank'], pos)} for j in np.argsort(d[i])[:3]]
    return out


# ---------------------------------------------------------------- history
def save_snapshot(scored):
    hdir = DATA / 'history'
    hdir.mkdir(exist_ok=True)
    (hdir / f'{TODAY.isoformat()}.json').write_text(json.dumps({r.pid: round(float(r.hit_pct_100), 2) for r in scored.itertuples()}))
    return [(p.stem, json.loads(p.read_text())) for p in sorted(hdir.glob('*.json'))]


def movers(history, scored, n=5):
    if len(history) < 2:
        return None
    (d0, a), (_, b) = history[-2], history[-1]
    rows = sorted(((b[r.pid] - a[r.pid], r) for r in scored.itertuples() if r.pid in a and r.pid in b), key=lambda x: x[0])
    item = lambda dlt, r: {'slug': r.slug, 'name': r.name, 'pos': r.pos, 'college': r.college_team, 'delta': round(dlt, 1)}
    return {'since': d0, 'up': [item(*x) for x in rows[::-1][:n] if x[0] > 0], 'down': [item(*x) for x in rows[:n] if x[0] < 0]}


# ---------------------------------------------------------------- exports
def export_board(scored):
    players = [{'s': r.slug, 'n': r.name, 'p': r.pos, 'c': r.college_team, 'cf': r.conference or '', 'r': int(r.rank),
                'h': r1(r.hit_pct_100), 'd': r1(r.p_draft_100), 'g': r1(r.p_hit_given_draft_100), 't': tiers_of(r), 'k': r.key_stats}
               for r in scored.sort_values('rank').itertuples()]
    dump(REPO / 'public' / 'data' / 'board.json', {'asof': ASOF, 'players': players})


def export_players(scored, drv, cps, history):
    hist_by = {}
    for d, snap in history:
        for pid, v in snap.items():
            hist_by.setdefault(pid, []).append([d, v])
    out = []
    for r in scored.itertuples():
        if r.hit_pct_100 < 0.5 and r.rank > 60:          # long shots get a board row but no page
            continue
        series, row = drv.get(r.pid, (None, None))
        out.append({'slug': r.slug, 'name': r.name, 'pos': r.pos, 'college': r.college_team, 'conf': r.conference or '',
                    'rank': int(r.rank), 'hit': r1(r.hit_pct_100), 'draft': r1(r.p_draft_100), 'hitIfDrafted': r1(r.p_hit_given_draft_100),
                    'score': int(r.model_score), 'tiers': tiers_of(r), 'keyStats': r.key_stats or '',
                    'drivers': driver_lists(series, row) if series is not None else None,
                    'comps': cps.get(r.pid, []), 'history': hist_by.get(r.pid, [])})
    dump(REPO / 'src' / 'data' / 'players.json', out)
    return out


def export_site(scored, players, mv):
    on_page = {p['slug'] for p in players}
    top = {}
    for pos in ['RB', 'WR', 'QB', 'TE']:
        rows = scored[scored.pos == pos].sort_values('hit_pct', ascending=False).head(5)
        top[pos] = [{'slug': r.slug, 'name': r.name, 'college': r.college_team, 'hit': r1(r.hit_pct_100), 'tiers': tiers_of(r)}
                    for r in rows.itertuples() if r.slug in on_page]
    dump(REPO / 'src' / 'data' / 'site.json', {'asof': ASOF, 'updated': TODAY.isoformat(), 'class': CLASS, 'top': top, 'movers': mv})


def export_track():
    val = pd.DataFrame(json.load(open(OUT / 'validation.json'))['rows'])
    bt = pd.DataFrame(json.load(open(OUT / 'backtest.json'))['rows'])
    lb = pd.DataFrame(json.load(open(OUT / 'lookback_summary.json'))['rows'])
    v = val[val.stage == 'base'].set_index('pos')
    acc = [{'pos': p, 'draftAuc': round(float(v.at[p, 'draft_auc']), 2), 'hitAuc': round(float(v.at[p, 'hit_auc']), 2),
            'topQ': round(float(v.at[p, 'drafted_top_q_hit'])), 'bottomQ': round(float(v.at[p, 'drafted_bottom_q_hit']))}
           for p in ['QB', 'RB', 'WR', 'TE']]
    item = lambda df: [{'name': r.name, 'pos': r.pos, 'college': r.college_team, 'year': int(r.draft_year), 'pick': int(r.pick),
                        'score': int(r.score_pct)} for r in df.itertuples()]
    calls = {'hits': item(bt[bt.hit == 1].sort_values('score_pct', ascending=False).head(8)),
             'misses': item(bt[bt.hit == 0].sort_values('score_pct', ascending=False).head(8)),
             'late': item(bt[(bt.hit == 1) & (bt.pick > 64)].sort_values('score_pct', ascending=False).head(8))}
    look = [{'pos': r.pos, 'drafted': int(r.drafted), 'model': round(float(r.model_corr), 2), 'post': round(float(r.post_corr), 2),
             'pick': round(float(r.pick_corr), 2)} for r in lb.itertuples()]
    dump(REPO / 'src' / 'data' / 'track.json', {'accuracy': acc, 'calls': calls, 'lookback': look})


def export_sleeper():
    lb = pd.DataFrame(json.load(open(OUT / 'lookback.json'))['rows'])
    try:
        ids = pd.read_csv(io.StringIO(urllib.request.urlopen(IDS_URL, timeout=60).read().decode()), dtype=str)
    except Exception:
        ids = pd.read_csv(sorted(glob.glob(str(ROOT.parent / '**/db_playerids.csv'), recursive=True))[0], dtype=str)
    pool = pd.read_pickle(DATA / 'pool_adv.pkl')[['player_id', 'gsis_id']].dropna()
    pool['id'] = pool.player_id.astype(str)
    ids['sleeper_id'] = ids.sleeper_id.str.replace(r'\.0$', '', regex=True)
    g = lb.merge(pool[['id', 'gsis_id']], on='id', how='left')
    g = g.merge(ids[['gsis_id', 'sleeper_id', 'team']].dropna(subset=['gsis_id', 'sleeper_id']), on='gsis_id', how='inner')
    g = g[(g.drafted == 1) | (g.signed_udfa == 1)]
    players = {}
    for r in g.itertuples():
        if r.post_hit_pct != r.post_hit_pct:
            continue
        t = [r1(getattr(r, f'post_tier{k + 1}')) for k in range(len(TIERS[r.pos]))] if r.pos in TIERS else None
        players[r.sleeper_id] = {'n': r.name, 'pos': r.pos, 'team': r.team if isinstance(r.team, str) else '',
                                 'pick': int(r.pick) if r.pick == r.pick else None, 'h': r1(r.post_hit_pct), 't': t}
    dump(REPO / 'public' / 'data' / 'sleeper_map.json', {'cls': CLASS - 1, 'players': players})
    return len(players)


def main():
    scored, fitted, pool = load()
    history = save_snapshot(scored)
    export_board(scored)
    players = export_players(scored, drivers(fitted, scored), comps(fitted, pool, scored), history)
    export_site(scored, players, movers(history, scored))
    export_track()
    n = export_sleeper()
    print(f'exported {len(players)} player pages, {n} rookies mapped to Sleeper, {len(history)} snapshot(s) -> {REPO}')


if __name__ == '__main__':
    main()
