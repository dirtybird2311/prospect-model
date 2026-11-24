"""Render the sample vertical short (1080x1920, 30 fps) from narration clips + timings.json.

    python3 narration/render_short.py narration/out/sample_rb1 out/short_rb1.mp4
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
BG, PANEL, INK, MUTED, GRID = (14, 17, 22), (23, 28, 36), (242, 244, 247), (139, 149, 165), (46, 54, 66)
ACCENT, ACCENT_DIM = (34, 197, 139), (30, 80, 64)
GF = '/usr/share/fonts/truetype/google-fonts/ofl'
ANTON = f'{GF}/anton/Anton-Regular.ttf'
DMS = f'{GF}/dmsans/DMSans[opsz,wght].ttf'
PAD = 0.35   # silence after each clip

RBS = [  # from out/prospects.json, Nov 24, 2026: odds of a top-12 (RB1) season in his first five, incl. draft odds
    {'rank': 1, 'name': 'Cole Tabb', 'school': 'Cincinnati', 'odds': 26.2},
    {'rank': 2, 'name': 'Jadan Baugh', 'school': 'Florida', 'odds': 21.2},
    {'rank': 3, 'name': "J'Koby Williams", 'school': 'Texas Tech', 'odds': 20.3},
]


def font(path, size, weight=None):
    f = ImageFont.truetype(path, size)
    if weight:
        try:
            f.set_variation_by_axes([14, weight])
        except Exception:
            pass
    return f


F = {'brand': font(ANTON, 64), 'tag': font(DMS, 34, 600), 'head': font(ANTON, 118), 'sub': font(DMS, 42, 500),
     'rank': font(ANTON, 72), 'name': font(DMS, 52, 700), 'school': font(DMS, 36, 500), 'pct': font(ANTON, 84),
     'cap': font(DMS, 44, 600), 'big': font(ANTON, 300), 'chip': font(DMS, 36, 600), 'url': font(ANTON, 120),
     'foot': font(DMS, 26, 500)}


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def fade(c, a, bg=BG):
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def chrome(d, caption, t):
    d.text((72, 96), 'STATABLE', font=F['brand'], fill=ACCENT)
    d.text((72 + d.textlength('STATABLE', font=F['brand']) + 24, 122), 'RB1 ODDS  ·  2027 CLASS', font=F['tag'], fill=MUTED)
    d.line((72, 196, W - 72, 196), fill=GRID, width=2)
    # caption panel
    a = ease(t / 0.25)
    lines = wrap(d, caption, F['cap'], W - 2 * 112)
    ph = 56 * len(lines) + 64
    y0 = 1690 - ph
    d.rounded_rectangle((72, y0, W - 72, 1690), radius=28, fill=fade(PANEL, a))
    for i, ln in enumerate(lines):
        d.text((112, y0 + 32 + 56 * i), ln, font=F['cap'], fill=fade(INK, a, PANEL))
    d.text((72, 1790), 'Statable model, Nov 24, 2026. Odds include draft odds. statableff.com', font=F['foot'], fill=MUTED)


def scene_hook(d, t):
    for i, ln in enumerate(['WHICH COLLEGE RBs', 'ARE MOST LIKELY TO', 'BECOME NFL RB1s?']):
        a = ease((t - 0.15 * i) / 0.4)
        d.text((72, 330 + i * 140 + (1 - a) * 40), ln, font=F['head'], fill=fade(ACCENT if i == 2 else INK, a))
    a = ease((t - 0.8) / 0.4)
    d.text((72, 800), 'Odds of at least one top-12 RB season', font=F['sub'], fill=fade(MUTED, a))
    d.text((72, 856), 'in his first five NFL years', font=F['sub'], fill=fade(MUTED, a))


def rows(d, t, shown, current, chip=None):
    x0, bar_x, bar_w = 72, 72, W - 144
    scale = (bar_w - 190) / 30.0   # 30% fills the track, leaving room for the label
    for r in RBS:
        if r['rank'] not in shown:
            continue
        y = 330 + (r['rank'] - 1) * 300
        is_cur = r['rank'] == current
        a = ease(t / 0.35) if is_cur else 1.0
        grow = ease((t - 0.15) / 0.7) if is_cur else 1.0
        ink = INK if is_cur else MUTED
        d.text((x0, y), f"#{r['rank']}", font=F['rank'], fill=fade(ACCENT if is_cur else MUTED, a))
        d.text((x0 + 130, y + 6), r['name'], font=F['name'], fill=fade(ink, a))
        d.text((x0 + 130, y + 70), r['school'], font=F['school'], fill=fade(MUTED, a))
        by = y + 140
        d.rounded_rectangle((bar_x, by, bar_x + bar_w, by + 56), radius=28, fill=PANEL)
        bw = max(56, r['odds'] * scale * grow)
        d.rounded_rectangle((bar_x, by, bar_x + bw, by + 56), radius=28, fill=ACCENT if is_cur else ACCENT_DIM)
        pct = f"{r['odds'] * grow:.0f}%"
        d.text((bar_x + bw + 20, by - 14), pct, font=F['pct'], fill=fade(ink, a))
    if chip:
        a = ease((t - 2.2) / 0.4)
        y = 330 + 3 * 300 - 10
        for i, c in enumerate(chip):
            tw = d.textlength(c, font=F['chip'])
            d.rounded_rectangle((72, y + i * 84, 72 + tw + 48, y + i * 84 + 64), radius=32, fill=fade(ACCENT_DIM, a))
            d.text((96, y + i * 84 + 12), c, font=F['chip'], fill=fade(INK, a))


def scene_context(d, t):
    a = ease(t / 0.4)
    d.text((72, 300), '1 IN 6', font=F['big'], fill=fade(ACCENT, a))
    d.text((72, 700), 'drafted running backs ever have', font=F['sub'], fill=fade(INK, a))
    d.text((72, 756), 'an RB1 season (2015-21 drafts)', font=F['sub'], fill=fade(INK, a))
    for i in range(6):
        b = ease((t - 0.3 - 0.12 * i) / 0.3)
        cx, cy, rr = 140 + i * 160, 1000, 58
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=fade(ACCENT if i == 0 else GRID, b))


def scene_cta(d, t):
    a = ease(t / 0.4)
    d.text((72, 400), 'FULL BOARD +', font=F['head'], fill=fade(INK, a))
    d.text((72, 540), 'TRACK RECORD', font=F['head'], fill=fade(INK, a))
    b = ease((t - 0.4) / 0.4)
    d.text((72, 760), 'statableff.com', font=F['url'], fill=fade(ACCENT, b))
    d.text((72, 950), 'Odds update every Wednesday', font=F['sub'], fill=fade(MUTED, b))


def draw(seg_id, t, caption):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    if seg_id == 'hook':
        scene_hook(d, t)
    elif seg_id == 'three':
        rows(d, t, {3}, 3)
    elif seg_id == 'two':
        rows(d, t, {2, 3}, 2)
    elif seg_id == 'one':
        rows(d, t, {1, 2, 3}, 1, chip=['131 scrimmage yds/game', '17% of team catches'])
    elif seg_id == 'context':
        scene_context(d, t)
    else:
        scene_cta(d, t)
    chrome(d, caption, t)
    return im


def render(clip_dir, out_path, stills=None):
    clip_dir, out_path = Path(clip_dir), Path(out_path)
    timings = json.loads((clip_dir / 'timings.json').read_text())
    if stills:   # preview frames only
        for s in timings:
            draw(s['id'], max(s['seconds'] - 0.2, 0.1), s['text']).save(Path(stills) / f"{s['id']}.png")
        return
    work = out_path.parent / f'.{out_path.stem}_parts'
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in timings:
        dur = s['seconds'] + PAD
        n = math.ceil(dur * FPS)
        tmp, fin = work / f"{s['id']}.tmp.mp4", work / f"{s['id']}.mp4"
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                              '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
                              '-pix_fmt', 'yuv420p', str(tmp)], stdin=subprocess.PIPE)
        for k in range(n):
            p.stdin.write(draw(s['id'], k / FPS, s['text']).tobytes())
        p.stdin.close()
        if p.wait() != 0:
            sys.exit(f"render failed: {s['id']}")
        os.replace(tmp, fin)
        parts.append((fin, clip_dir / f"{s['id']}.wav", n / FPS))
    # audio: each clip padded to its scene length, then concatenated
    alist = work / 'audio.txt'
    with open(alist, 'w') as f:
        for fin, wav, dur in parts:
            padded = work / f'{wav.stem}.pad.wav'
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav), '-af', f'apad=whole_dur={dur}',
                            '-ar', '24000', '-ac', '1', str(padded)], check=True)
            f.write(f"file '{padded.name}'\n")
    vlist = work / 'video.txt'
    vlist.write_text(''.join(f"file '{fin.name}'\n" for fin, _, _ in parts))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(vlist), '-f', 'concat',
                    '-safe', '0', '-i', str(alist), '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
                    '-b:a', '160k', '-movflags', '+faststart', str(out_path)], check=True)
    print('wrote', out_path)


if __name__ == '__main__':
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
