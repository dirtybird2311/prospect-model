"""Download raw college and NFL data. Re-downloads the current season every run."""
import urllib.request
from .config import DATA, CFB_RAW, NFLVERSE, STATS_SEASONS, ROSTER_SEASONS, NFL_STAT_SEASONS, PROSPECT_SEASON


def _get(url, dest, refresh=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not refresh:
        return
    tmp = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    print('downloaded', dest.name, flush=True)


def fetch_all():
    for yr in STATS_SEASONS:
        _get(f'{CFB_RAW}/player_stats/csv/player_stats_{yr}.csv', DATA / 'cfb' / f'player_stats_{yr}.csv',
             refresh=(yr == PROSPECT_SEASON))
    for yr in ROSTER_SEASONS:
        _get(f'{CFB_RAW}/rosters/csv/cfb_rosters_{yr}.csv', DATA / 'cfb' / f'cfb_rosters_{yr}.csv',
             refresh=(yr == PROSPECT_SEASON))
    for f in ['draft_picks/draft_picks.csv', 'combine/combine.csv', 'players/players.csv']:
        _get(f'{NFLVERSE}/{f}', DATA / 'nfl' / f.split('/')[-1], refresh=True)
    for yr in NFL_STAT_SEASONS:
        _get(f'{NFLVERSE}/stats_player/stats_player_reg_{yr}.csv', DATA / 'nfl' / f'stats_{yr}.csv')
    # current NFL season so far, for the lookback at the latest draft class
    try:
        _get(f'{NFLVERSE}/stats_player/stats_player_reg_{PROSPECT_SEASON}.csv', DATA / 'nfl' / f'stats_{PROSPECT_SEASON}.csv', refresh=True)
    except Exception as e:
        print('no current-season NFL stats yet:', e)
