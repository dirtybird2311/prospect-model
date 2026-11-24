# Prospect model

Odds that a college QB, RB, WR or TE becomes a fantasy starter in the NFL, built from college stats.

**Hit % = P(drafted) × P(hit | drafted)**, where a hit is a top-24 RB/WR or top-12 QB/TE half-PPR season within a player's first five NFL seasons. The model score is the player's percentile of hit % among current prospects at his position.

Results feed the **2027 Prospect Odds** dashboard: https://claude.ai/artifact/KUxBuq4qcafCGfizwZy7oJ

## Run it

```
pip install -r requirements.txt
python3 run.py            # downloads data (recruiting CSVs come from the GitHub Action) (~1 GB first time), fits, scores, writes out/*.json
python3 run.py --no-fetch # reuse downloaded files
```

Outputs in `out/`: `prospects.json`, `validation.json`, `weights.json`, `backtest.json`. Each is `{rows, meta}`; `meta` carries the source code that produced it.

## Data

- College play-by-play player stats and rosters: [sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data) (stats from 2014; rosters from 2009 to estimate enrollment year). Updated weekly during the season. Some bowl and playoff games are missing.
- NFL draft picks, combine, player season stats: [nflverse](https://github.com/nflverse/nflverse-data).

## Models

| Model | Trained on | Target |
|---|---|---|
| Draft odds | Every FBS QB/RB/WR/TE with a real role whose college career ended 2014–2025 | Drafted or not |
| Hit odds | Players drafted 2015–2021 (full 5-year window) | Top-N season within 5 years |

Draft odds: per-position L2 logistic regression. Hit odds: RB uses L2 logistic regression on hit/miss; QB, WR and TE use ridge regression on log best-season points relative to the hit line (RB24/WR24/QB12/TE12), calibrated to hit odds, and also train on 2022–2024 draftees using their best season so far (validated on 2015–2021 only). Regularization chosen by leave-one-draft-class-out AUC. Final hit % is recalibrated (Platt scaling) only where that improves the Brier score by 2%+, currently TE after the combine.

**Inputs:** per-game production, team shares (yards, catches, TDs), efficiency, best season share, breakout timing, conference strength (drafted skill players per team-season), estimated age at draft (from first roster year; ~0.6 yr mean error vs real birth dates), redshirt status, height, weight.

**Conference adjustment (WR/TE):** production × conference strength. **Transfer flag (WR/TE):** played for more than one school. **Recruiting (`pipeline/recruiting.py`):** 247Sports Composite rating, pulled from CollegeFootballData.com by the "Pull recruiting rankings" GitHub Action (needs a `CFBD_API_KEY` repository secret) into `data/recruiting/`. Matched by CFBD athlete id, else name + school + class year. Used in the RB draft and hit models and the WR hit model (with an unrated flag); no help for QB, TE or post-draft. **Advanced stats (from play-by-play, `pipeline/advanced.py`):** RB rushing success rate and first downs per carry (hit model); TE receiving yards/game vs Power defenses (both models). Explosive rate, 3rd-down and red-zone share, yards per target and competitive-game production were tested and left out.

**Combine stage:** once a prospect has combine results (nflverse updates in early March), he's scored with models that add combine invite and athletic testing. 

**Post-draft stage:** once a player is drafted, hit odds come from draft slot plus his college hit score (QB, RB) or draft slot alone (WR, TE, where the college score added nothing on top of draft slot). Used by the lookback, and for the board after each April draft. Tested and left out: explosive-run rate, drafted RB teammates, committee-back flags, landing spot (`pipeline/landing.py`: vacated opportunity, team offense) and real birth dates. Also tested and left out elsewhere: QB athletic testing.

**Fantasy tiers (RB/WR):** odds a player's best season in his first five reaches the top 12 (RB1/WR1), 13–24 (RB2/WR2) or 25–36 (flex) in a 12-team league; one model per line (RB: logistic; WR: the points model with its own calibration). Backtest AUC, top 12/24/36: RB .81/.78/.76, WR .70/.71/.69.

**Undrafted signings:** undrafted players who signed with an NFL team (from nflverse's player table, by ESPN id or name + school) get post-draft odds as if picked 300th. Only 7 of 530 such QB/RB/WR/TE signees from 2015–2021 hit, so their odds stay near that base rate.

**Lookback (`pipeline/lookback.py`):** scores the most recent draft class with models refit without it, next to its draft slots and current-season half-PPR results.

## Validation (leave-one-draft-class-out)

Run `python3 run.py` to reprint. As of Nov 23, 2026:

| Pos | Draft AUC | Hit AUC (drafted) | Top vs bottom quarter of drafted players |
|---|---|---|---|
| QB | 0.91 | 0.79 (0.81 with combine) | 53% vs 5% |
| RB | 0.94 | 0.78 (0.77 with combine) | 60% vs 3% |
| WR | 0.91 | 0.71 (0.71 with combine) | 36% vs 8% |
| TE | 0.87 | 0.68 (0.76 with combine) | 38% vs 4% |

Post-draft hit accuracy (drafted 2015–2021, nested leave-one-class-out): QB 0.90 (draft slot alone 0.89), RB 0.85 (0.83), WR 0.83, TE 0.86.

## Known limits

- College stats can't see film, injuries, character or scheme fit.
- Draft odds treat the current season as a player's last; underclassmen who return to school aren't in that draft.
- Players whose final season was outside FBS, or who transferred, may be missed.
