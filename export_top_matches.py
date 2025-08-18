#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from argparse import ArgumentParser
from datetime import datetime

PATTERN = re.compile(r"person_(\d+)_.+\.jpg$")


def collect_latest_person_images(src_dir: str):
    # Use detection_summary.png mtime as anchor for latest run
    summary = os.path.join(src_dir, 'detection_summary.png')
    if not os.path.isfile(summary):
        # fallback: use most recently modified person_* files
        files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.startswith('person_') and f.endswith('.jpg')]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files
    anchor_time = os.path.getmtime(summary)
    # gather person files within +/- 90 seconds of anchor
    files = []
    for f in os.listdir(src_dir):
        if not f.startswith('person_') or not f.endswith('.jpg'):
            continue
        fp = os.path.join(src_dir, f)
        try:
            mt = os.path.getmtime(fp)
            if abs(mt - anchor_time) <= 90:
                files.append(fp)
        except Exception:
            continue
    # sort by index in filename ascending
    def index_key(path: str):
        m = PATTERN.search(os.path.basename(path))
        return int(m.group(1)) if m else 999999
    files.sort(key=index_key)
    return files


def build_html(out_dir: str, title: str, images):
    html = []
    html.append('<!DOCTYPE html>')
    html.append('<html lang="en"><head>')
    html.append('<meta charset="utf-8"/>')
    html.append('<meta name="viewport" content="width=device-width, initial-scale=1"/>')
    html.append('<title>Top Matches</title>')
    html.append('<style> body { font-family: "Times New Roman", Times, serif; margin:20px; background:none;} h1{margin:6px 0 16px;} .grid{display:flex;flex-wrap:wrap;gap:12px;} .card{border:1px solid #ddd;padding:8px;} img{max-width:480px;height:auto;display:block;} .name{color:#333;font-size:12px;margin-top:4px;} </style>')
    html.append('</head><body>')
    html.append(f'<h1>{title}</h1>')
    html.append('<div class="grid">')
    for img in images:
        name = os.path.basename(img)
        html.append('<div class="card">')
        html.append(f'<img src="{name}" />')
        html.append(f'<div class="name">{name}</div>')
        html.append('</div>')
    html.append('</div>')
    html.append('</body></html>')
    with open(os.path.join(out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))


def main():
    ap = ArgumentParser(description='Export Top-N person images from latest run into a new folder with HTML index')
    ap.add_argument('--source', default='outputs/all_detections', help='source directory of person_*.jpg and detection_summary.png')
    ap.add_argument('--dest', default='', help='destination directory (optional). If empty, auto-generate.')
    ap.add_argument('--topn', type=int, default=10, help='number of top matches to export')
    ap.add_argument('--tag', default='query', help='tag for naming output folder, e.g., red_helmet')
    args = ap.parse_args()

    src = os.path.abspath(args.source)
    images = collect_latest_person_images(src)
    if not images:
        print('No person images found in latest run.')
        return
    topn = max(1, min(args.topn, len(images)))
    images = images[:topn]

    if args.dest:
        out_dir = os.path.abspath(args.dest)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        out_dir = os.path.join(os.path.dirname(src), f'{args.tag}_top{topn}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    copied = []
    for p in images:
        name = os.path.basename(p)
        dst = os.path.join(out_dir, name)
        shutil.copy2(p, dst)
        copied.append(dst)

    build_html(out_dir, f'Top {topn} - {args.tag}', copied)
    print('Saved to:', out_dir)


if __name__ == '__main__':
    main() 