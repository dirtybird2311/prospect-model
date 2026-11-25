"""Narrate a script with Google Cloud Text-to-Speech (Chirp 3 HD voices): one WAV per segment + timings.json.

    GOOGLE_TTS_API_KEY=... python3 narration/narrate_google.py narration/scripts/x.json narration/out/x

Standard library only. The key comes from the GOOGLE_TTS_API_KEY secret; restrict it to the Text-to-Speech API.
Free tier: 1 million characters a month on Chirp 3 HD voices (a 30-second short is about 600 characters).
"""
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

API = 'https://texttospeech.googleapis.com/v1/text:synthesize'
SR = 24000


def synth(text, voice, rate, key):
    body = {'input': {'text': text},
            'voice': {'languageCode': voice[:5], 'name': voice},
            'audioConfig': {'audioEncoding': 'LINEAR16', 'sampleRateHertz': SR}}
    if rate and rate != 1.0:
        body['audioConfig']['speakingRate'] = rate
    req = urllib.request.Request(f'{API}?key={key}', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return base64.b64decode(json.load(r)['audioContent'])
    except urllib.error.HTTPError as e:
        msg = e.read().decode()[:300]
        if 'speakingRate' in body['audioConfig'] and e.code == 400:
            print(f'{voice}: speaking rate not accepted ({msg}); using normal speed')
            return synth(text, voice, 1.0, key)
        sys.exit(f'Google TTS error {e.code} for {voice}: {msg}')


def main(script_path, out_dir):
    key = os.environ.get('GOOGLE_TTS_API_KEY')
    if not key:
        sys.exit('Set GOOGLE_TTS_API_KEY.')
    spec = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    variants = spec.get('variants') or [{'tag': '', 'voice': spec['voice'], 'speed': spec.get('speed', 1.0)}]
    for v in variants:
        vout = out / v['tag'] if v['tag'] else out
        vout.mkdir(parents=True, exist_ok=True)
        timings = []
        for seg in spec['segments']:
            wav_bytes = synth(seg['text'], v['voice'], v.get('speed', 1.0), key)
            (vout / f"{seg['id']}.wav").write_bytes(wav_bytes)
            with wave.open(io.BytesIO(wav_bytes)) as w:
                secs = w.getnframes() / w.getframerate()
            timings.append({'id': seg['id'], 'seconds': round(secs, 3), 'text': seg['text']})
            print(v['tag'], seg['id'], timings[-1]['seconds'])
        (vout / 'timings.json').write_text(json.dumps(timings, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
