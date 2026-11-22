# Prospect model

Odds that a college QB, RB, WR or TE becomes a fantasy starter in the NFL, built from college stats.

**Hit % = P(drafted) × P(hit | drafted)**, where a hit is a top-24 RB/WR or top-12 QB/TE half-PPR season within a player's first five NFL seasons. The model score is the player's percentile of hit % among current prospects at his position.

Results feed the **2027 Prospect Odds** dashboard: https://claude.ai/artifact/KUxBuq4qcafCGfizwZy7oJ

## Run it

```
pip install -r requirements.txt
python3 run.py            # downloads data (~1 GB first time), fits, scores, writes out/*.json
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

Both are per-position L2 logistic regressions on standardized inputs, regularization chosen by leave-one-draft-class-out AUC.

**Inputs:** per-game production, team shares (yards, catches, TDs), efficiency, best season share, breakout timing, conference strength (drafted skill players per team-season), estimated age at draft (from first roster year; ~0.6 yr mean error vs real birth dates), redshirt status, height, weight.

**Conference adjustment (WR/TE):** production × conference strength. **Advanced stats (from play-by-play, `pipeline/advanced.py`):** RB rushing success rate and first downs per carry (hit model); TE receiving yards/game vs Power defenses (both models). Explosive rate, 3rd-down and red-zone share, yards per target and competitive-game production were tested and left out.

**Combine stage:** once a prospect has combine results (nflverse updates in early March), he's scored with models that add combine invite and athletic testing. No recruiting rankings: the data source doesn't include them.

## Validation (leave-one-draft-class-out)

Run `python3 run.py` to reprint. As of Nov 21, 2026:

| Pos | Draft AUC | Hit AUC (drafted) | Top vs bottom quarter of drafted players |
|---|---|---|---|
| QB | 0.91 | 0.75 | 47% vs 5% |
| RB | 0.93 | 0.76 | 57% vs 6% |
| WR | 0.91 | 0.63 | 28% vs 9% |
| TE | 0.88 | 0.61 (0.77 with combine) | 38% vs 5% |

## Known limits

- College stats can't see film, injuries, character or scheme fit.
- Draft odds treat the current season as a player's last; underclassmen who return to school aren't in that draft.
- Players whose final season was outside FBS, or who transferred, may be missed.
