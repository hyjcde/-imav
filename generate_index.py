#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from glob import glob
from typing import List, Dict, Any, Optional
from datetime import datetime


def discover_latest_results(base_dir: str, limit: int = 4) -> List[str]:
    roots = [base_dir, os.path.join(base_dir, 'outputs')]
    dirs: List[str] = []
    for root in roots:
        pattern = os.path.join(root, 'advanced_results_*')
        dirs.extend([d for d in glob(pattern) if os.path.isdir(d)])
    # 按修改时间排序，最新在前
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs[:limit]


def read_report(result_dir: str) -> Optional[Dict[str, Any]]:
    report_path = os.path.join(result_dir, 'advanced_analysis_report.json')
    if not os.path.isfile(report_path):
        return None
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def find_match_image_by_frame(result_dir: str, frame_number: int) -> Optional[str]:
    # Prefer images that explicitly include the frame number in filename
    # Example: advanced_match_01_frame_2100_score_0.381.jpg
    pattern = os.path.join(result_dir, f'advanced_match_*_frame_{frame_number}_score_*.jpg')
    files = glob(pattern)
    if files:
        # If multiple, pick the highest rank (smallest number after advanced_match_)
        def rank_key(path: str) -> int:
            m = re.search(r'advanced_match_(\d+)_frame_', os.path.basename(path))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return 999999
            return 999999
        files.sort(key=rank_key)
        return files[0]
    # Fallback: any advanced_match containing this frame number
    alt_pattern = os.path.join(result_dir, f'advanced_match_*{frame_number}*.jpg')
    alt_files = glob(alt_pattern)
    if alt_files:
        return sorted(alt_files)[0]
    return None


def build_section_for_dir(base_dir: str, result_dir: str, limit_top: int = 10) -> str:
    report = read_report(result_dir)
    rel_dir = os.path.relpath(result_dir, start=base_dir)
    title = rel_dir
    description = ''
    total_matches = None
    if report and isinstance(report, dict):
        info = report.get('analysis_info', {})
        description = info.get('description', '')
        total_matches = info.get('total_matches', None)
    # Collect top matches from report if available
    top_matches: List[Dict[str, Any]] = []
    if report and isinstance(report, dict):
        best = report.get('best_matches', [])
        if isinstance(best, list):
            for item in best[:limit_top]:
                try:
                    rank = item.get('rank')
                    frame_number = item.get('frame_number')
                    total_score = item.get('total_score')
                    explanation = item.get('explanation', [])
                    top_matches.append({
                        'rank': rank,
                        'frame_number': frame_number,
                        'total_score': total_score,
                        'explanation': explanation,
                    })
                except Exception:
                    continue
    # Fallback if report missing: infer from filenames
    if not top_matches:
        files = glob(os.path.join(result_dir, 'advanced_match_*_frame_*_score_*.jpg'))
        def file_rank_key(path: str) -> int:
            m = re.search(r'advanced_match_(\d+)_frame_', os.path.basename(path))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return 999999
            return 999999
        files.sort(key=file_rank_key)
        for p in files[:limit_top]:
            m = re.search(r'_frame_(\d+)_', os.path.basename(p))
            frame_num = int(m.group(1)) if m else -1
            top_matches.append({
                'rank': file_rank_key(p),
                'frame_number': frame_num,
                'total_score': None,
                'explanation': [],
            })

    # Build HTML
    html = []
    html.append(f'<section style="margin-bottom: 36px;">')
    html.append(f'<h2 style="margin: 8px 0;">{title}</h2>')
    if description:
        html.append(f'<div style="margin: 6px 0; color:#333;"><b>Prompt:</b> {description}</div>')
    if total_matches is not None:
        html.append(f'<div style="margin: 6px 0; color:#555;">Total detections: {total_matches}</div>')

    # Table header
    html.append('<table style="border-collapse: collapse; width: 100%;">')
    html.append('<thead>\n<tr>\n'
                '<th style="text-align:left;padding:6px;border-bottom:1px solid #ddd;">Rank</th>'
                '<th style="text-align:left;padding:6px;border-bottom:1px solid #ddd;">Frame</th>'
                '<th style="text-align:left;padding:6px;border-bottom:1px solid #ddd;">Match (detailed)</th>'
                '<th style="text-align:left;padding:6px;border-bottom:1px solid #ddd;">All persons</th>'
                '<th style="text-align:left;padding:6px;border-bottom:1px solid #ddd;">Score / Notes</th>'
                '</tr>\n</thead>')
    html.append('<tbody>')

    for m in top_matches:
        frame = m.get('frame_number')
        rank = m.get('rank')
        score = m.get('total_score')
        notes = ''
        if score is not None:
            notes += f'total_score: {score:.3f}'
        exp_list = m.get('explanation') or []
        if exp_list:
            if notes:
                notes += ' | '
            notes += ' ; '.join([str(x) for x in exp_list])

        match_img = find_match_image_by_frame(result_dir, frame) if frame is not None else None
        all_persons_img = os.path.join(result_dir, f'all_persons_frame_{frame}.jpg') if frame is not None else None
        match_rel = os.path.relpath(match_img, start=base_dir) if match_img and os.path.isfile(match_img) else ''
        persons_rel = os.path.relpath(all_persons_img, start=base_dir) if all_persons_img and os.path.isfile(all_persons_img) else ''

        # Row
        html.append('<tr>')
        html.append(f'<td style="vertical-align:top;padding:6px;border-bottom:1px solid #f0f0f0;">{rank if rank is not None else "-"}</td>')
        html.append(f'<td style="vertical-align:top;padding:6px;border-bottom:1px solid #f0f0f0;">{frame if frame is not None else "-"}</td>')

        # Match image
        if match_rel:
            html.append('<td style="vertical-align:top;padding:6px;border-bottom:1px solid #f0f0f0;">'
                        f'<a href="{match_rel}" target="_blank"><img src="{match_rel}" style="max-width:480px; height:auto; border:1px solid #ddd;"/></a>'
                        f'<div style="color:#666; font-size:12px;">{os.path.basename(match_rel)}</div>'
                        '</td>')
        else:
            html.append('<td style="vertical-align:top;padding:6px;border-bottom:1px solid #f0f0f0; color:#999;">(missing)</td>')

        # All persons image
        if persons_rel:
            html.append('<td style="vertical-align:top;padding:6px;border-bottom:1px solid #f0f0f0;">'
                        f'<a href="{persons_rel}" target="_blank"><img src="{persons_rel}" style="max-width:420px; height:auto; border:1px solid #ddd;"/></a>'
                        f'<div style="color:#666; font-size:12px;">{os.path.basename(persons_rel)}</div>'
                        '</td>')
        else:
            html.append('<td style="vertical-align:top;padding:6px;border-bottom:1px solid #f0f0f0; color:#999;">(missing)</td>')

        html.append(f'<td style="vertical-align:top;padding:6px;border-bottom:1px solid #f0f0f0; color:#333;">{notes}</td>')
        html.append('</tr>')

    html.append('</tbody></table>')
    html.append('</section>')
    return '\n'.join(html)


def main():
    parser = argparse.ArgumentParser(description='Generate an HTML index for advanced results with Top-N matches and all-persons frames.')
    parser.add_argument('dirs', nargs='*', help='advanced_results directories (optional). If omitted, auto-detect latest 4.')
    parser.add_argument('-o', '--output', default='', help='Output HTML path (default: advanced_overview_YYYYMMDD_HHMM.html in base dir)')
    parser.add_argument('--limit', type=int, default=10, help='Top-N matches per directory (default: 10)')
    args = parser.parse_args()

    base_dir = os.getcwd()

    result_dirs: List[str] = []
    if args.dirs:
        for d in args.dirs:
            if os.path.isdir(d):
                result_dirs.append(os.path.abspath(d))
    else:
        result_dirs = discover_latest_results(base_dir, limit=4)

    if not result_dirs:
        print('No result directories found.')
        return

    # Build HTML
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = args.output.strip() or os.path.join(base_dir, f'advanced_overview_{now_str}.html')

    html_parts: List[str] = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html lang="en"><head>')
    html_parts.append('<meta charset="utf-8"/>')
    html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1"/>')
    html_parts.append('<title>Advanced Results Overview</title>')
    html_parts.append('<style> body { font-family: "Times New Roman", Times, serif; margin: 20px; background: none; } h1{margin: 6px 0 16px;} h2{font-size:18px;} .hint{color:#666;} </style>')
    html_parts.append('</head><body>')
    html_parts.append('<h1>Advanced Results Overview</h1>')
    html_parts.append('<div class="hint">Generated at ' + datetime.now().strftime('%Y-%m-%d %H:%M') + '</div>')

    for d in result_dirs:
        html_parts.append(build_section_for_dir(base_dir, d, limit_top=args.limit))

    html_parts.append('</body></html>')

    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    print('Saved:', output_path)


if __name__ == '__main__':
    main() 