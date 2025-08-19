#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from typing import List, Dict, Any, Tuple

from drone_optimized_detector import DroneOptimizedPersonFinder
from utils.clip_utils import load_clip, compute_clip_similarity


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def expand_prompts(text: str) -> List[str]:
    """基础提示扩展，不针对特定颜色或形状"""
    t = text.strip()
    if not t:
        return []
    
    variants = {t}
    # 通用扩展
    variants.add(f"a person {t}")
    variants.add(f"person with {t}")
    variants.add(f"person wearing {t}")
    
    return list(variants)


def draw_overlay(frame: np.ndarray, bbox: tuple, title: str) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    img = frame.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    t_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x1, max(0, y1 - t_size[1] - 8)), (x1 + t_size[0] + 6, y1), (0, 255, 0), -1)
    cv2.putText(img, title, (x1 + 3, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def build_tracks(items: List[Tuple[int, Dict[str, Any]]], max_dist: float = 50.0, min_len: int = 2) -> Dict[int, Dict[str, Any]]:
    """
    简易跨帧连贯性跟踪：按中心点最近邻链接，返回 track_id -> {length, mean_score, dets}
    """
    frames = {}
    for fn, det in items:
        frames.setdefault(fn, []).append(det)
    sorted_frames = sorted(frames.keys())
    next_id = 1
    tracks: Dict[int, Dict[str, Any]] = {}
    last_layer = []
    
    for fi, fn in enumerate(sorted_frames):
        layer = []
        dets = frames[fn]
        for det in dets:
            cx, cy = det.get('center', (0, 0))
            best_tid, best_d = -1, 1e9
            for (tid, (px, py)) in last_layer:
                d = np.hypot(cx - px, cy - py)
                if d < best_d:
                    best_d, best_tid = d, tid
            if best_d <= max_dist and best_tid in tracks:
                tr = tracks[best_tid]
                tr['length'] += 1
                tr['sum_score'] += float(det.get('clip_score', 0.0))
                tr['dets'].append((fn, det))
                layer.append((best_tid, (cx, cy)))
            else:
                tid = next_id; next_id += 1
                tracks[tid] = {'length': 1, 'sum_score': float(det.get('clip_score', 0.0)), 'dets': [(fn, det)]}
                layer.append((tid, (cx, cy)))
        last_layer = layer
    
    for tid in list(tracks.keys()):
        tr = tracks[tid]
        tr['mean_score'] = tr['sum_score'] / max(1, tr['length'])
        if tr['length'] < min_len:
            tr['mean_score'] *= 0.5
    return tracks


def main():
    ap = ArgumentParser(description='Pure YOLO detection + CLIP semantic understanding for text-to-person retrieval')
    ap.add_argument('video', help='video path')
    ap.add_argument('text', help='query text, e.g., red helmet, blue jacket, yellow vest')
    ap.add_argument('--max-frames', type=int, default=12, help='frames to sample')
    ap.add_argument('--topn', type=int, default=10, help='export top-N matches')
    ap.add_argument('--conf', type=float, default=0.20)
    ap.add_argument('--iou', type=float, default=0.35)
    ap.add_argument('--min-size', type=int, default=10)
    ap.add_argument('--tile', type=int, default=960)
    ap.add_argument('--overlap', type=float, default=0.40)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--tta', action='store_true', help='enable multi-size+offset+flip TTA')
    ap.add_argument('--frames', type=str, default='', help='optional comma-separated frame numbers to keep')
    ap.add_argument('--track-w', type=float, default=0.2, help='weight of track consistency [0-1]')
    ap.add_argument('--track-max-dist', type=float, default=60.0, help='max pixel distance to link detections')
    ap.add_argument('--track-min-len', type=int, default=2, help='min length of track to avoid penalty')
    args = ap.parse_args()

    # 1) 检测所有人（高召回）
    finder = DroneOptimizedPersonFinder(
        min_person_size=args.min_size,
        confidence_threshold=args.conf,
        nms_threshold=args.iou,
        tile_size=args.tile,
        tile_overlap=args.overlap,
        tile_imgsz=args.imgsz
    )
    setattr(finder, 'use_tta', bool(args.tta))
    det_results = finder.find_all_persons_in_video(args.video, args.max_frames)
    if det_results.get('status') != 'success':
        print('Detection failed.')
        return

    restrict_frames = set()
    if args.frames.strip():
        try:
            restrict_frames = set(int(x) for x in args.frames.split(',') if x.strip())
        except Exception:
            restrict_frames = set()

    persons: List[Dict[str, Any]] = []
    frame_map: Dict[int, np.ndarray] = {}
    for finfo in det_results['frame_info']:
        if restrict_frames and finfo['frame_number'] not in restrict_frames:
            continue
        frame_map[finfo['frame_number']] = finfo['frame']
        for det in finfo['detections']:
            persons.append({
                'frame_number': finfo['frame_number'],
                **det
            })

    if not persons:
        print('No persons found.')
        return

    # 2) 纯CLIP语义理解
    clip_ctx = load_clip()
    if clip_ctx is None:
        print('CLIP backend not available. Please ensure open-clip-torch or openai-clip installed.')
        return
    
    prompts = expand_prompts(args.text) or [args.text]

    scored = []
    for det in persons:
        crop = det.get('person_crop')
        if crop is None or crop.size == 0:
            continue
        
        # 纯CLIP语义分数
        sims = []
        for p in prompts:
            s = compute_clip_similarity(clip_ctx, crop, p)
            if s is not None:
                sims.append(s)
        clip_score = float(max(sims)) if sims else 0.0
        
        det['clip_score'] = clip_score
        det['final_score'] = clip_score  # 直接使用CLIP分数
        scored.append((clip_score, det))

    if not scored:
        print('No scores computed.')
        return

    # 3) 可选跨帧跟踪重打分
    if args.track_w > 0.0:
        items = [(d['frame_number'], d) for _, d in scored]
        tracks = build_tracks(items, max_dist=args.track_max_dist, min_len=args.track_min_len)
        
        det2final = []
        for clip_score, det in scored:
            best_track_score = 0.0
            for tid, tr in tracks.items():
                if any(id(det) == id(dd) for (_, dd) in tr['dets']):
                    length_norm = min(1.0, tr['length'] / max(2, args.track_min_len + 2))
                    track_score = 0.6 * length_norm + 0.4 * float(tr['mean_score'])
                    best_track_score = track_score
                    break
            
            final_score = (1.0 - args.track_w) * clip_score + args.track_w * best_track_score
            det['track_score'] = best_track_score
            det['final_score'] = final_score
            det2final.append((final_score, det))
        scored = det2final
    else:
        scored = [(d[0], {**d[1], 'final_score': d[0], 'track_score': 0.0}) for d in scored]

    scored.sort(key=lambda x: x[0], reverse=True)

    # 输出
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = os.path.join('outputs', f'pure_retrieval_{ts}')
    ensure_dir(out_dir)

    # 每帧合成图显示纯CLIP分数
    per_frame = {}
    for final, det in scored:
        fn = det['frame_number']
        per_frame.setdefault(fn, []).append((final, det))

    for fn, items in per_frame.items():
        frame = frame_map.get(fn)
        if frame is None:
            continue
        composite = frame.copy()
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        for rank_in_frame, (s, d) in enumerate(items_sorted, start=1):
            x1, y1, x2, y2 = d['bbox']
            color = (0, 0, 255) if rank_in_frame == 1 else (0, 255, 0)
            thickness = 3 if rank_in_frame == 1 else 2
            cv2.rectangle(composite, (x1, y1), (x2, y2), color, thickness)
            
            # 简化标签：只显示CLIP分数和可选跟踪分数
            if args.track_w > 0.0:
                label = f"#{rank_in_frame} CLIP:{d.get('clip_score',0.0):.3f} Track:{d.get('track_score',0.0):.3f}"
            else:
                label = f"#{rank_in_frame} CLIP:{d.get('clip_score',0.0):.3f}"
                
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(composite, (x1, max(0, y1 - t_size[1] - 4)), (x1 + t_size[0] + 6, y1), color, -1)
            cv2.putText(composite, label, (x1 + 3, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(out_dir, f'frame_{fn}_pure_clip.jpg'), composite)

    # Top-N结果
    topn = min(args.topn, len(scored))
    for rank, (final, det) in enumerate(scored[:topn], start=1):
        fn = det['frame_number']
        frame = frame_map.get(fn)
        x1, y1, x2, y2 = det['bbox']
        overlay = draw_overlay(frame, (x1, y1, x2, y2), f"#{rank} CLIP:{final:.3f} F{fn}")
        cv2.imwrite(os.path.join(out_dir, f'rank_{rank:02d}_frame_{fn}_clip_{final:.3f}.jpg'), overlay)
        crop = det.get('person_crop')
        if crop is not None and crop.size > 0:
            cv2.imwrite(os.path.join(out_dir, f'rank_{rank:02d}_crop_clip_{final:.3f}.jpg'), crop)

    print('Saved:', out_dir)
    print(f'Pure CLIP-based retrieval completed.')
    print(f'Top-1 CLIP score: {scored[0][0]:.3f}' if scored else 'No matches found.')


if __name__ == '__main__':
    main() 