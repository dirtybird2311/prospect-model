"""Pull high-school recruiting rankings (247Sports Composite, via CollegeFootballData.com) into data/recruiting/.

    CFBD_API_KEY=... python3 scripts/fetch_recruiting.py --first 2010 --last 2026

One request per class year (17 requests for 2010-2026; the free CFBD tier allows 1,000 a month).
Standard library only, so it runs on a bare GitHub Actions runner.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

API = 'https://api.collegefootballdata.com/recruiting/players'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'recruiting'
FIELDS = ['year', 'ranking', 'name', 'position', 'stars', 'rating', 'school', 'committedTo', 'height', 'weight',
          'city', 'stateProvince', 'athleteId', 'id', 'recruitType']


def pull(year, key):
    req = urllib.request.Request(f'{API}?year={year}&classification=HighSchool',
                                 headers={'Authorization': f'Bearer {key}', 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--first', type=int, default=2010)
    ap.add_argument('--last', type=int, default=2026)
    a = ap.parse_args()
    key = os.environ.get('CFBD_API_KEY')
    if not key:
        sys.exit('Set CFBD_API_KEY (a free key from https://collegefootballdata.com/key).')
    OUT.mkdir(parents=True, exist_ok=True)
    for y in range(a.first, a.last + 1):
        rows = pull(y, key)
        with open(OUT / f'recruits_{y}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in FIELDS})
        print(f'{y}: {len(rows)} recruits')
        time.sleep(1)


if __name__ == '__main__':
    main()
