"""Build the NFL lines file (spread, total, win odds) for upcoming games and save a dated snapshot.

    python3 scripts/fetch_lines.py

Source order:
  1. Pinnacle's lines as shown on OddsPortal, read by the OddsHarvester step of the GitHub workflow
     (data/lines/oddsportal_raw.json). This script only parses that file.
  2. nflverse's schedule file (consensus spread and total) for any game OddsPortal didn't give us.

Writes data/lines/lines_<UTC timestamp>.csv (one snapshot per pull, so line movement is kept),
data/lines/latest.csv and data/lines/status.txt. Columns: season, week, gameday, kickoff_utc, away_team,
home_team, spread_home (negative = home favored), total, home_implied, away_implied, home_win_prob
(no-vig, Pinnacle only), source. Lines are model inputs only; they are never published on the site.
"""
import csv
import glob
import io
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'data' / 'lines'
NFLVERSE = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
TEAMS = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
}
FIELDS = ['season', 'week', 'gameday', 'kickoff_utc', 'away_team', 'home_team', 'spread_home', 'total',
          'home_implied', 'away_implied', 'home_win_prob', 'source']


def get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': '*/*', **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')


def implied(spread_home, total):
    home = total / 2 - spread_home / 2
    return round(home, 2), round(total - home, 2)


def to_dec(v):
    """A price as decimal odds; accepts decimal (1.91), American (-110, +120) or fractional (10/11)."""
    v = str(v).strip()
    if '/' in v:
        a, b = v.split('/')
        return 1 + float(a) / float(b)
    x = float(v)
    if abs(x) >= 100:
        return 1 + (x / 100 if x > 0 else 100 / -x)
    return x


def no_vig(p1, p2):
    """Two prices -> no-vig probability of the first side."""
    q1, q2 = 1 / to_dec(p1), 1 / to_dec(p2)
    return q1 / (q1 + q2)


# ---------------------------------------------------------------- nflverse (schedule, consensus lines)
def schedule():
    rows = list(csv.DictReader(io.StringIO(get(NFLVERSE, timeout=60))))
    return rows


def upcoming(rows, days=10):
    now = datetime.now(timezone.utc).date()
    out = []
    for g in rows:
        try:
            d = datetime.strptime(g['gameday'], '%Y-%m-%d').date()
        except ValueError:
            continue
        if now <= d <= now + timedelta(days=days) and not g['home_score']:
            out.append(g)
    return out


def nflverse_lines(games):
    out = []
    for g in games:
        if not g['spread_line'] or not g['total_line']:
            continue
        spread_home = -float(g['spread_line'])        # nflverse spread_line > 0 means the home team is favored
        total = float(g['total_line'])
        h, a = implied(spread_home, total)
        out.append({'season': g['season'], 'week': g['week'], 'gameday': g['gameday'], 'kickoff_utc': '',
                    'away_team': g['away_team'], 'home_team': g['home_team'], 'spread_home': spread_home,
                    'total': total, 'home_implied': h, 'away_implied': a, 'home_win_prob': '', 'source': 'nflverse'})
    return out


# ---------------------------------------------------------------- OddsPortal (Pinnacle)
LINE_RE = re.compile(r'(over_under|asian_handicap)_(-?\d+(?:_\d+)?)')


def _line(token):
    sign = -1 if token.startswith('-') else 1
    parts = token.lstrip('-').split('_')
    return sign * float(parts[0] + ('.' + parts[1] if len(parts) > 1 else ''))


def _pin(entries):
    """The Pinnacle row of one market's bookmaker list."""
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and 'pinnacle' in str(e.get('bookmaker_name', e.get('bookmaker', ''))).lower():
            return e
    return None


def _closest_even(cands):
    """Of (line, p_first_side) candidates, the line priced closest to 50/50 (the main line)."""
    return min(cands, key=lambda c: abs(c[1] - 0.5))[0] if cands else None


def oddsportal_lines(sched):
    files = sorted(glob.glob(str(OUT / 'oddsportal_raw*.json')))
    if not files:
        raise RuntimeError('no OddsPortal file from the harvester step')
    data = json.load(open(files[-1]))
    matches = data if isinstance(data, list) else data.get('matches', data.get('data', []))
    by_teams = {(s['away_team'], s['home_team']): s for s in sched}
    out = []
    for m in matches:
        home, away = TEAMS.get(m.get('home_team', '')), TEAMS.get(m.get('away_team', ''))
        if not home or not away:
            continue
        totals, spreads, win = [], [], ''
        for k, v in m.items():
            pin = _pin(v) if k.endswith('_market') else None
            if not pin:
                continue
            if k.startswith('home_away'):
                win = round(no_vig(pin['1'], pin['2']), 4)
                continue
            lm = LINE_RE.match(k)
            if not lm:
                continue
            line = _line(lm.group(2))
            if lm.group(1) == 'over_under':
                totals.append((line, no_vig(pin['odds_over'], pin['odds_under'])))
            else:
                spreads.append((line, no_vig(pin['1'], pin['2'])))
        total, spread_home = _closest_even(totals), _closest_even(spreads)
        if total is None or spread_home is None:
            continue
        s = by_teams.get((away, home), {})
        h, a = implied(spread_home, total)
        out.append({'season': s.get('season', ''), 'week': s.get('week', ''), 'gameday': s.get('gameday', ''),
                    'kickoff_utc': m.get('match_date', ''), 'away_team': away, 'home_team': home,
                    'spread_home': spread_home, 'total': total, 'home_implied': h, 'away_implied': a,
                    'home_win_prob': win, 'source': 'pinnacle (oddsportal)'})
    return out


def main():
    sched = upcoming(schedule())
    rows, note = [], ''
    try:
        rows = oddsportal_lines(sched)
        note = f'oddsportal/pinnacle: {len(rows)} games'
    except Exception as e:
        note = f'oddsportal failed ({type(e).__name__}: {e}); using nflverse'
    have = {(r['away_team'], r['home_team']) for r in rows}
    fill = [r for r in nflverse_lines(sched) if (r['away_team'], r['home_team']) not in have]
    rows += fill
    rows.sort(key=lambda r: (str(r['gameday']), r['home_team']))
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%MZ')
    for path in (OUT / f'lines_{stamp}.csv', OUT / 'latest.csv'):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    (OUT / 'status.txt').write_text(f'{stamp} {note}; {len(fill)} games from nflverse\n')
    print(f'{note}; {len(fill)} games from nflverse; {len(rows)} total -> {OUT}/lines_{stamp}.csv')
    for r in rows:
        print(f"  {r['gameday']} {r['away_team']:>3} @ {r['home_team']:<3} spread {r['spread_home']:+5.1f} total {r['total']:4.1f}  [{r['source']}]")


if __name__ == '__main__':
    sys.exit(main())
