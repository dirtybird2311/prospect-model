"""Run the whole pipeline: download data, build features, fit and validate models, score prospects, export.

    python3 run.py            # full run (downloads only what's missing, always refreshes the current season)
    python3 run.py --no-fetch # reuse downloaded files
"""
import sys
import pickle
from datetime import datetime, timezone

from pipeline.config import DATA, OUT
from pipeline.fetch import fetch_all
from pipeline.college import build_college
from pipeline.features import build_pools
from pipeline.advanced import build_advanced, attach
from pipeline.models import fit_all, fit_post, score, weights
import pandas as pd
from pipeline.export import export, export_lookback
from pipeline.lookback import lookback
from pipeline.config import PROSPECT_SEASON


def main():
    if '--no-fetch' not in sys.argv:
        fetch_all()
    cs, _ = build_college()
    pool, pros, conf = build_pools(cs)
    adv = build_advanced()
    pool, pros = attach(pool, adv), attach(pros, adv)
    fitted, val, back = fit_all(pool)
    post, post_val = fit_post(pool, fitted)
    print(post_val.to_string(index=False))
    val = pd.concat([val, post_val], ignore_index=True)
    scored = score(fitted, pros)
    OUT.mkdir(exist_ok=True)
    with open(DATA / 'fitted.pkl', 'wb') as f:
        pickle.dump((fitted, post), f)
    pool.to_pickle(DATA / 'pool_adv.pkl')
    print(val[val.stage != 'post'][['pos', 'stage', 'draft_auc', 'hit_auc', 'overall_auc', 'drafted_top_q_hit', 'drafted_bottom_q_hit']].to_string(index=False))
    export(scored, val, back, weights(fitted), datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
    sc, summ = lookback(pool, PROSPECT_SEASON)
    if sc is not None:
        export_lookback(sc, summ, PROSPECT_SEASON, datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
        print(summ.to_string(index=False))
    print('done ->', OUT)


if __name__ == '__main__':
    main()
