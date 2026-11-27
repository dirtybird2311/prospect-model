"""Pull FanDuel and Caesars NFL player props from SportsGameOdds and save a dated snapshot.

    SGO_API_KEY=... python3 scripts/fetch_props.py

Free SportsGameOdds plan: 10 requests a minute, 2,500 objects a month, so this pulls only games in the next
8 days. Writes data/props/props_<UTC timestamp>.csv, data/props/latest.csv and data/props/status.txt, one row per
player, stat and book:
    event_id, kickoff_utc, away_team, home_team, player_id, player, stat, period, bet_type, book,
    line, over_odds, under_odds, p_over_novig
plus a consensus row per player and stat (book = 'consensus': mean line, mean no-vig over probability,
FanDuel and Caesars weighted equally). Props are a check on our projections, never published as odds.
"""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'data' / 'props'
API = 'https://api.sportsgameodds.com/v2/events'
BOOKS = ['fanduel', 'caesars']
DAYS = 8                                          # games kicking off within this many days
FIELDS = ['event_id', 'kickoff_utc', 'away_team', 'home_team', 'player_id', 'player', 'stat', 'period', 'bet_type',
          'book', 'line', 'over_odds', 'under_odds', 'p_over_novig']
TEAM_SIDES = {'home', 'away', 'all'}


def get(params, key):
    url = API + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'x-api-key': key, 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def dec(american):
    a = float(american)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def no_vig(over, under):
    q1, q2 = 1 / dec(over), 1 / dec(under)
    return q1 / (q1 + q2)


def team_name(ev, side):
    t = (ev.get('teams') or {}).get(side) or {}
    names = t.get('names') or {}
    return names.get('short') or names.get('medium') or t.get('teamID', '')


def player_name(ev, pid):
    p = (ev.get('players') or {}).get(pid) or {}
    if p.get('name'):
        return p['name']
    if p.get('firstName') or p.get('lastName'):
        return f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
    return pid.rsplit('_', 2)[0].replace('_', ' ').title()       # JAMES_COOK_1_NFL -> James Cook


def events(key):
    out, cursor = [], None
    for _ in range(10):                                           # pages; stops early at the date window
        params = {'leagueID': 'NFL', 'oddsAvailable': 'true', 'bookmakerID': ','.join(BOOKS), 'limit': 25}
        if cursor:
            params['cursor'] = cursor
        j = get(params, key)
        if not j.get('success', True):
            raise RuntimeError(j.get('error', 'request failed'))
        batch = j.get('data') or []
        out += batch
        cursor = j.get('nextCursor')
        last = (batch[-1].get('status') or {}).get('startsAt', '') if batch else ''
        if not cursor or not batch or last > (datetime.now(timezone.utc) + timedelta(days=DAYS)).isoformat():
            break
        time.sleep(7)                                             # stay under 10 requests a minute
    return out


def rows_for(ev):
    starts = (ev.get('status') or {}).get('startsAt', '')
    base = {'event_id': ev.get('eventID', ''), 'kickoff_utc': starts,
            'away_team': team_name(ev, 'away'), 'home_team': team_name(ev, 'home')}
    pairs = {}
    for odd_id, o in (ev.get('odds') or {}).items():
        parts = odd_id.split('-')
        if len(parts) != 5:
            continue
        stat, ent, period, bet, side = parts
        if ent in TEAM_SIDES or bet not in ('ou', 'yn'):
            continue
        for book, b in ((o.get('byBookmaker') or {}).items()):
            if book not in BOOKS or not b.get('available', True) or b.get('odds') in (None, ''):
                continue
            k = (ent, stat, period, bet, book)
            rec = pairs.setdefault(k, {})
            rec[side] = b['odds']
            if b.get('overUnder') not in (None, ''):
                rec['line'] = b['overUnder']
    rows = []
    for (ent, stat, period, bet, book), rec in pairs.items():
        over, under = rec.get('over', rec.get('yes')), rec.get('under', rec.get('no'))
        if over is None:
            continue
        p = round(no_vig(over, under), 4) if under is not None else ''
        rows.append({**base, 'player_id': ent, 'player': player_name(ev, ent), 'stat': stat, 'period': period,
                     'bet_type': bet, 'book': book, 'line': rec.get('line', ''), 'over_odds': over,
                     'under_odds': under if under is not None else '', 'p_over_novig': p})
    return rows


def consensus(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['event_id'], r['player_id'], r['stat'], r['period'], r['bet_type']), []).append(r)
    out = []
    for g in groups.values():
        lines = [float(r['line']) for r in g if r['line'] != '']
        probs = [r['p_over_novig'] for r in g if r['p_over_novig'] != '']
        if len(g) < 2 or not probs:
            continue
        out.append({**{k: g[0][k] for k in FIELDS[:9]}, 'book': 'consensus',
                    'line': round(sum(lines) / len(lines), 2) if lines else '', 'over_odds': '', 'under_odds': '',
                    'p_over_novig': round(sum(probs) / len(probs), 4)})
    return out


def main():
    key = os.environ.get('SGO_API_KEY')
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%MZ')
    if not key:
        (OUT / 'status.txt').write_text(f'{stamp} no SGO_API_KEY secret; nothing pulled\n')
        print('No SGO_API_KEY set; skipping.')
        return 0
    try:
        evs = events(key)
    except urllib.error.HTTPError as e:
        msg = f'{stamp} SportsGameOdds refused ({e.code}: {e.read()[:200]!r})'
        (OUT / 'status.txt').write_text(msg + '\n')
        print(msg)
        return 0
    horizon = (datetime.now(timezone.utc) + timedelta(days=DAYS)).isoformat()
    evs = [e for e in evs if (e.get('status') or {}).get('startsAt', '') <= horizon]
    if evs and not (OUT / 'sample_event.json').exists():                     # one raw event, to check the format
        (OUT / 'sample_event.json').write_text(json.dumps(evs[0], indent=1)[:200000])
    rows = [r for e in evs for r in rows_for(e)]
    rows += consensus(rows)
    rows.sort(key=lambda r: (r['kickoff_utc'], r['player'], r['stat'], r['book']))
    for path in (OUT / f'props_{stamp}.csv', OUT / 'latest.csv'):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    per_book = {b: sum(r['book'] == b for r in rows) for b in BOOKS + ['consensus']}
    note = f'{stamp} {len(evs)} games, {len(rows)} rows ' + ', '.join(f'{b} {n}' for b, n in per_book.items())
    (OUT / 'status.txt').write_text(note + '\n')
    print(note)
    return 0


if __name__ == '__main__':
    sys.exit(main())

