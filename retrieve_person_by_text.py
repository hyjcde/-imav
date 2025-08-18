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
    t = text.strip()
    if not t:
        return []
    # 简单扩展（中英混合）
    variants = {t}
    low = t.lower()
    if '红色' in t or 'red' in low:
        variants.add(f"{t} wearing red")
        variants.add("red helmet")
        variants.add("a person wearing a red helmet")
    if '头盔' in t or 'helmet' in low:
        variants.add("wearing helmet")
        variants.add("construction helmet")
        variants.add("hard hat")
    if '黄色' in t or 'yellow' in low:
        variants.add(f"{t} wearing yellow")
        variants.add("yellow vest")
    return list(variants)


def draw_overlay(frame: np.ndarray, bbox: tuple, title: str) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    img = frame.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    t_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x1, max(0, y1 - t_size[1] - 8)), (x1 + t_size[0] + 6, y1), (0, 255, 0), -1)
    cv2.putText(img, title, (x1 + 3, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def head_color_ratio_bgr(person_crop: np.ndarray, color: str = 'red') -> float:
    if person_crop is None or person_crop.size == 0:
        return 0.0
    h, w = person_crop.shape[:2]
    if h < 6 or w < 6:
        return 0.0
    head = person_crop[: max(2, h // 3), :]
    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 0.0
    # HSV ranges
    if color == 'red':
        ranges = [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))]
    elif color == 'yellow':
        ranges = [((20, 120, 70), (35, 255, 255))]
    else:
        return 0.0
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in ranges:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask_total = cv2.bitwise_or(mask_total, mask)
    cnt = int(cv2.countNonZero(mask_total))
    return float(cnt) / float(total)


def head_shape_score_bgr(person_crop: np.ndarray) -> float:
    """估计头部形状圆度得分 [0,1]。优先对红色区域做圆度；无红色时以整体边缘近似。"""
    if person_crop is None or person_crop.size == 0:
        return 0.0
    h, w = person_crop.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    head = person_crop[: max(3, h // 3), :]
    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    # red mask
    r1_lo, r1_hi = (0, 120, 70), (10, 255, 255)
    r2_lo, r2_hi = (170, 120, 70), (180, 255, 255)
    mask1 = cv2.inRange(hsv, np.array(r1_lo), np.array(r1_hi))
    mask2 = cv2.inRange(hsv, np.array(r2_lo), np.array(r2_hi))
    rmask = cv2.bitwise_or(mask1, mask2)
    def circularity_from_mask(mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        peri = float(cv2.arcLength(c, True))
        if peri <= 1e-3 or area <= 1.0:
            return 0.0
        circ = 4.0 * np.pi * area / (peri * peri)
        return float(max(0.0, min(1.0, circ)))
    circ = circularity_from_mask(rmask)
    if circ > 0.0:
        # 强化阈值映射（>0.6认为较圆）
        return float(min(1.0, max(0.0, (circ - 0.4) / 0.4)))
    # fallback: 用整体边缘近似圆度
    gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    return circularity_from_mask(edges)


def build_tracks(items: List[Tuple[int, Dict[str, Any]]], max_dist: float = 50.0, min_len: int = 2) -> Dict[int, Dict[str, Any]]:
    """
    简易跨帧连贯性跟踪：按中心点最近邻链接，返回 track_id -> {length, mean_score, dets}
    items: [(frame_number, det_dict)] 按帧序升序
    det_dict 需包含 'center' 与 'fused_score'
    """
    # 按帧聚类
    frames = {}
    for fn, det in items:
        frames.setdefault(fn, []).append(det)
    sorted_frames = sorted(frames.keys())
    next_id = 1
    tracks: Dict[int, Dict[str, Any]] = {}
    last_layer = []  # [(track_id, center)]
    for fi, fn in enumerate(sorted_frames):
        layer = []
        dets = frames[fn]
        used = set()
        for det in dets:
            cx, cy = det.get('center', (0, 0))
            # 找到最近的上层轨迹
            best_tid, best_d = -1, 1e9
            for (tid, (px, py)) in last_layer:
                d = np.hypot(cx - px, cy - py)
                if d < best_d:
                    best_d, best_tid = d, tid
            if best_d <= max_dist and best_tid in tracks:
                # 追加
                tr = tracks[best_tid]
                tr['length'] += 1
                tr['sum_score'] += float(det.get('fused_score', 0.0))
                tr['dets'].append((fn, det))
                layer.append((best_tid, (cx, cy)))
            else:
                # 新轨迹
                tid = next_id; next_id += 1
                tracks[tid] = {'length': 1, 'sum_score': float(det.get('fused_score', 0.0)), 'dets': [(fn, det)]}
                layer.append((tid, (cx, cy)))
        last_layer = layer
    # 计算均值分数，过滤过短轨迹
    for tid in list(tracks.keys()):
        tr = tracks[tid]
        tr['mean_score'] = tr['sum_score'] / max(1, tr['length'])
        if tr['length'] < min_len:
            tr['mean_score'] *= 0.5  # 过短降权
    return tracks


def main():
    ap = ArgumentParser(description='Two-stage: detect all persons, then rank crops by CLIP similarity to text')
    ap.add_argument('video', help='video path')
    ap.add_argument('text', help='query text, e.g., 红色头盔 / red helmet')
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
    ap.add_argument('--color-w', type=float, default=0.4, help='weight of color prior when applicable [0-1]')
    ap.add_argument('--shape-w', type=float, default=0.3, help='weight of head shape prior when applicable [0-1]')
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

    # 2) CLIP 语义相似度 + 先验
    clip_ctx = load_clip()
    if clip_ctx is None:
        print('CLIP backend not available. Please ensure open-clip-torch or openai-clip installed.')
        return
    prompts = expand_prompts(args.text) or [args.text]

    want_helmet = (('头盔' in args.text) or ('helmet' in args.text.lower()))
    want_red = (('红色' in args.text) or ('red' in args.text.lower()))

    scored = []
    for det in persons:
        crop = det.get('person_crop')
        if crop is None or crop.size == 0:
            continue
        # CLIP score
        sims = []
        for p in prompts:
            s = compute_clip_similarity(clip_ctx, crop, p)
            if s is not None:
                sims.append(s)
        clip_score = float(max(sims)) if sims else 0.0
        # Color prior (head red/yellow)
        color_score = 0.0
        if want_red:
            ratio = head_color_ratio_bgr(crop, 'red')
            color_score = min(1.0, max(0.0, (ratio - 0.02) / 0.08))
        # Shape prior (helmet-like roundness)
        shape_score = head_shape_score_bgr(crop) if want_helmet else 0.0
        # Fuse
        fused = clip_score
        if want_red:
            fused = (1.0 - args.color_w) * fused + args.color_w * color_score
        if want_helmet:
            fused = (1.0 - args.shape_w) * fused + args.shape_w * shape_score
        det['clip_score'] = clip_score
        det['color_score'] = color_score
        det['shape_score'] = shape_score
        det['fused_score'] = fused
        scored.append((fused, det))

    if not scored:
        print('No scores computed.')
        return

    # 3) 跨帧跟踪重打分
    if args.track_w > 0.0:
        items = [(d['frame_number'], d) for _, d in scored]
        tracks = build_tracks(items, max_dist=args.track_max_dist, min_len=args.track_min_len)
        # 计算每个检测的轨迹分数（长度/均分综合）
        det2final = []
        for fused, det in scored:
            # 找到所属轨迹
            best_tid = None
            best_score = 0.0
            for tid, tr in tracks.items():
                # 粗匹配：是否在该轨迹的 dets 列表中
                if any(id(det) == id(dd) for (_, dd) in tr['dets']):
                    # 轨迹分 = 归一化长度*(0.6) + 均值得分*(0.4)
                    length_norm = min(1.0, tr['length'] / max(2, args.track_min_len + 2))
                    track_score = 0.6 * length_norm + 0.4 * float(tr['mean_score'])
                    best_tid = tid; best_score = track_score
                    break
            final_score = (1.0 - args.track_w) * fused + args.track_w * best_score
            det['track_score'] = best_score
            det['final_score'] = final_score
            det2final.append((final_score, det))
        scored = det2final
    else:
        # 没有跟踪时，final= fused
        scored = [(d[0], {**d[1], 'final_score': d[0], 'track_score': 0.0}) for d in scored]

    scored.sort(key=lambda x: x[0], reverse=True)

    # 输出
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = os.path.join('outputs', f'retrieval_{ts}')
    ensure_dir(out_dir)

    # 每帧合成图显示 clip/color/shape/track/final
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
            label = f"#{rank_in_frame} fin:{d.get('final_score',0.0):.3f}/c:{d.get('clip_score',0.0):.2f}/h:{d.get('color_score',0.0):.2f}/s:{d.get('shape_score',0.0):.2f}/t:{d.get('track_score',0.0):.2f}"
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            cv2.rectangle(composite, (x1, max(0, y1 - t_size[1] - 4)), (x1 + t_size[0] + 6, y1), color, -1)
            cv2.putText(composite, label, (x1 + 3, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(out_dir, f'frame_{fn}_all_scores.jpg'), composite)

    topn = min(args.topn, len(scored))
    for rank, (final, det) in enumerate(scored[:topn], start=1):
        fn = det['frame_number']
        frame = frame_map.get(fn)
        x1, y1, x2, y2 = det['bbox']
        overlay = draw_overlay(frame, (x1, y1, x2, y2), f"#{rank} final:{final:.3f} F{fn}")
        cv2.imwrite(os.path.join(out_dir, f'rank_{rank:02d}_frame_{fn}_final_{final:.3f}.jpg'), overlay)
        crop = det.get('person_crop')
        if crop is not None and crop.size > 0:
            cv2.imwrite(os.path.join(out_dir, f'rank_{rank:02d}_crop_final_{final:.3f}.jpg'), crop)

    print('Saved:', out_dir)


if __name__ == '__main__':
    main() 