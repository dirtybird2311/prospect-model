"""Pull NFL game lines (spread and total) for upcoming games and save a dated snapshot.

    python3 scripts/fetch_lines.py

Source order:
  1. Pinnacle, from the public JSON feed its own website reads (guest.api.arcadia.pinnacle.com).
     Pinnacle blocks some regions; if the feed refuses or fails, we do not route around it.
  2. nflverse's schedule file (consensus spread and total), which is always the fallback.

Writes data/lines/lines_<UTC timestamp>.csv (one snapshot per pull, so line movement is kept) and
data/lines/latest.csv. Columns: season, week, gameday, kickoff_utc, away_team, home_team, spread_home
(negative = home favored), total, home_implied, away_implied, home_win_prob (no-vig, Pinnacle only), source.
Lines are model inputs only; they are never published on the site.
"""
import csv
import io
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'data' / 'lines'
NFLVERSE = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
PIN_SITE = 'https://www.pinnacle.com/en/football/nfl/matchups/'
PIN_API = 'https://guest.api.arcadia.pinnacle.com/0.1'
NFL_LEAGUE = 889
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


def no_vig(p1, p2):
    """Two American prices -> no-vig probability of the first side."""
    dec = lambda a: 1 + (a / 100 if a > 0 else 100 / -a)
    q1, q2 = 1 / dec(p1), 1 / dec(p2)
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


# ---------------------------------------------------------------- Pinnacle (public website feed)
KEY_RE = re.compile(r'x-api-key["\']?\s*[:,=]\s*["\']([A-Za-z0-9]{20,})["\']', re.I)
DIAG = []


def pinnacle_key():
    """The feed key the website itself sends; found in the page or its scripts (and the chunks they load)."""
    page = get(PIN_SITE)
    m = KEY_RE.search(page)
    if m:
        return m.group(1)
    seen, queue = set(), re.findall(r'(?:src|href)=["\']?([^"\'\s>]+\.js)', page)
    DIAG.append(f'page {len(page)} chars, {len(queue)} scripts, arcadia in page: {"arcadia" in page}')
    while queue and len(seen) < 40:
        s = queue.pop(0)
        url = s if s.startswith('http') else 'https://www.pinnacle.com' + ('' if s.startswith('/') else '/') + s
        if url in seen:
            continue
        seen.add(url)
        try:
            js = get(url)
        except Exception:
            continue
        m = KEY_RE.search(js)
        if m:
            return m.group(1)
        if 'arcadia' in js:
            DIAG.append(f'arcadia mentioned in {url.rsplit("/", 1)[-1]}')
        base = url.rsplit('/', 1)[0]
        queue += [c if c.startswith('http') or c.startswith('/') else base + '/' + c
                  for c in re.findall(r'["\']([\w./-]+\.js)["\']', js)][:30]
    raise RuntimeError(f'no feed key found ({len(seen)} scripts read; ' + '; '.join(DIAG[:5]) + ')')


def pinnacle_lines(sched):
    key = pinnacle_key()
    hdr = {'x-api-key': key, 'Referer': 'https://www.pinnacle.com/', 'Origin': 'https://www.pinnacle.com',
           'Accept': 'application/json'}
    matchups = json.loads(get(f'{PIN_API}/leagues/{NFL_LEAGUE}/matchups', hdr))
    markets = json.loads(get(f'{PIN_API}/leagues/{NFL_LEAGUE}/markets/straight', hdr))
    games = {}
    for m in matchups:
        if m.get('type') != 'matchup' or m.get('parentId') or len(m.get('participants', [])) != 2:
            continue
        side = {p['alignment']: TEAMS.get(p['name']) for p in m['participants']}
        if not side.get('home') or not side.get('away'):
            continue
        games[m['id']] = {'home': side['home'], 'away': side['away'], 'start': m.get('startTime', '')}
    lines = {}
    for mk in markets:
        g = games.get(mk.get('matchupId'))
        if not g or mk.get('period') != 0 or mk.get('isAlternate') or mk.get('status') not in (None, 'open'):
            continue
        prices = {p.get('designation'): p for p in mk.get('prices', [])}
        rec = lines.setdefault(mk['matchupId'], {})
        if mk.get('type') == 'spread' and 'home' in prices:
            rec['spread_home'] = float(prices['home']['points'])
        elif mk.get('type') == 'total' and 'over' in prices:
            rec['total'] = float(prices['over']['points'])
        elif mk.get('type') == 'moneyline' and 'home' in prices and 'away' in prices:
            rec['home_win_prob'] = round(no_vig(prices['home']['price'], prices['away']['price']), 4)
    by_teams = {(s['away_team'], s['home_team']): s for s in sched}
    out = []
    for mid, rec in lines.items():
        g = games[mid]
        if 'spread_home' not in rec or 'total' not in rec:
            continue
        s = by_teams.get((g['away'], g['home']), {})
        h, a = implied(rec['spread_home'], rec['total'])
        out.append({'season': s.get('season', ''), 'week': s.get('week', ''), 'gameday': s.get('gameday', g['start'][:10]),
                    'kickoff_utc': g['start'], 'away_team': g['away'], 'home_team': g['home'],
                    'spread_home': rec['spread_home'], 'total': rec['total'], 'home_implied': h, 'away_implied': a,
                    'home_win_prob': rec.get('home_win_prob', ''), 'source': 'pinnacle'})
    return out


def main():
    sched = upcoming(schedule())
    rows, note = [], ''
    try:
        rows = pinnacle_lines(sched)
        note = f'pinnacle: {len(rows)} games'
    except urllib.error.HTTPError as e:
        note = f'pinnacle refused ({e.code}); using nflverse'
    except Exception as e:                              # any other failure: fall back, never retry around a block
        note = f'pinnacle failed ({type(e).__name__}: {e}); using nflverse'
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
    (OUT / 'status.txt').write_text(f'{stamp} {note}\n')
    print(f'{note}; {len(fill)} games from nflverse; {len(rows)} total -> {OUT}/lines_{stamp}.csv')
    for r in rows:
        print(f"  {r['gameday']} {r['away_team']:>3} @ {r['home_team']:<3} spread {r['spread_home']:+5.1f} total {r['total']:4.1f}  [{r['source']}]")


if __name__ == '__main__':
    sys.exit(main())
