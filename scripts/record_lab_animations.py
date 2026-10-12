#!/usr/bin/env python3
"""Record the README animations from docs/lab/lab3d.html (the page itself is not modified).

The page is opened in headless Chrome, set to Inference + Paper view, and its own
timeline slider is moved frame by frame. Each clip first shows the whole-volume
encoding and the first units slowly, then fast-forwards to the end.

    pip install playwright && python scripts/record_lab_animations.py --chrome /usr/bin/google-chrome
Requires ffmpeg.
"""
from __future__ import annotations
import argparse
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PE = 0.06                       # share of the lab timeline used by the first (encoding / window grid) phase
CLIPS = {                        # name: (rows, head, units on the timeline)
    'core_ae_both': ('both', 'liif', 343),
    'dense_ae': ('dense', 'liif', 343),
    'core_ae_liif': ('ours', 'liif', 512),
    'core_ae_clif': ('ours', 'clif', 216),
}


def timeline(units, intro=18, slow_units=3, per_unit=14, rush=44, hold=10):
    """Timeline positions in [0, 1] for each frame."""
    ps = [PE * i / intro for i in range(intro)]
    unit = (1 - PE) / units
    ps += [PE + unit * k / per_unit for k in range(slow_units * per_unit)]
    start = PE + unit * slow_units
    ps += [start + (0.9999 - start) * (i / (rush - 1)) ** 2 for i in range(rush)]
    return ps + [0.9999] * hold


def record(page, name, rows, head, units, frames_dir: Path):
    for group, value in (('c-mode', 'infer'), ('c-view', 'paper'), ('c-rows', rows), ('c-head', head)):
        page.click(f'#{group} button[data-v="{value}"]')
    page.wait_for_timeout(400)
    stage = page.locator('#stage')
    for i, p in enumerate(timeline(units)):
        page.evaluate("""v => { const r = document.getElementById('c-time');
            r.value = Math.round(v * 10000); r.dispatchEvent(new Event('input')); }""", p)
        page.wait_for_timeout(60)
        stage.screenshot(path=str(frames_dir / f'{name}_{i:04d}.png'))


def encode(frames_dir: Path, name: str, out: Path, fps: int, width: int):
    src = str(frames_dir / f'{name}_%04d.png')
    scale = f'scale={width}:-2:flags=lanczos'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', src,
                    '-vf', scale, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / f'{name}.mp4')], check=True)
    palette = frames_dir / f'{name}_palette.png'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-vf', f'{scale},palettegen=max_colors=128',
                    str(palette)], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', src, '-i', str(palette),
                    '-lavfi', f'{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4', str(out / f'{name}.gif')],
                   check=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--chrome', help='Chrome/Chromium executable (default: Playwright browser)')
    a.add_argument('--out', default=str(ROOT / 'docs' / 'media'))
    a.add_argument('--fps', type=int, default=12)
    a.add_argument('--width', type=int, default=960)
    a.add_argument('--clips', nargs='*', default=list(CLIPS))
    args = a.parse_args()
    from playwright.sync_api import sync_playwright
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chrome, args=['--use-gl=angle', '--use-angle=swiftshader',
                                                                       '--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1500, 'height': 1000}, device_scale_factor=1.5)
        page.goto((ROOT / 'docs' / 'lab' / 'lab3d.html').as_uri())
        page.wait_for_function('document.querySelectorAll(".lab").length > 10', timeout=60000)
        page.wait_for_timeout(1500)
        for name in args.clips:
            rows, head, units = CLIPS[name]
            record(page, name, rows, head, units, Path(tmp))
            encode(Path(tmp), name, out, args.fps, args.width)
            print('wrote', out / f'{name}.gif')
        browser.close()


if __name__ == '__main__':
    main()
