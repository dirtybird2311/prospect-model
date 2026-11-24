"""Narrate a script with Kokoro (open-source, Apache-2.0 voice model): one WAV per segment + timings.json.

    python3 narration/narrate.py narration/scripts/sample_rb1.json narration/out/sample_rb1
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SR = 24000


def main(script_path, out_dir):
    spec = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code='a')   # American English
    timings = []
    for seg in spec['segments']:
        audio = np.concatenate([np.asarray(a) for _, _, a in pipe(seg['text'], voice=spec.get('voice', 'af_heart'),
                                                                speed=spec.get('speed', 1.0))])
        sf.write(out / f"{seg['id']}.wav", audio, SR)
        timings.append({'id': seg['id'], 'seconds': round(len(audio) / SR, 3), 'text': seg['text']})
        print(seg['id'], timings[-1]['seconds'])
    (out / 'timings.json').write_text(json.dumps(timings, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
