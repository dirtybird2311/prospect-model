"""Shared settings for the prospect model pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'          # downloaded raw files (not committed)
OUT = ROOT / 'out'            # model outputs and dashboard files (not committed)

CFB_RAW = 'https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main'
NFLVERSE = 'https://github.com/nflverse/nflverse-data/releases/download'

STATS_SEASONS = range(2014, 2027)     # college play-level stats available from 2014
ROSTER_SEASONS = range(2009, 2027)    # earlier rosters only to estimate enrollment year (age)
NFL_STAT_SEASONS = range(2015, 2026)  # completed NFL seasons used for outcomes
PROSPECT_SEASON = 2026                # current college season
DRAFT_YEAR = PROSPECT_SEASON + 1

POS = ['QB', 'RB', 'WR', 'TE']
HIT_WINDOW = 5                         # top-N season within first 5 NFL seasons
HIT_CUT = {'QB': 12, 'RB': 24, 'WR': 24, 'TE': 12}
POWER = {'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12', 'FBS Independents'}
FBS = POWER | {'American Athletic', 'Mountain West', 'Sun Belt', 'Mid-American', 'Conference USA'}

# minimum usage in a player's final season to enter the pools
MIN_USAGE = {'QB': ('pass_att', 50), 'RB': ('rush_att', 30), 'WR': ('rec', 10), 'TE': ('rec', 8)}
