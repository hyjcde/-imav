#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from ultralytics import YOLO


def _init_yolo(model_name: str = 'weights/yolov8m.pt') -> YOLO:
    import torch
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = patched_load
    try:
        model = YOLO(model_name)
        return model
    finally:
        torch.load = original_load


def detect_persons_in_frame(model: YOLO, frame: np.ndarray,
                            conf_th: float = 0.18,
                            iou_th: float = 0.35,
                            min_size: int = 10) -> List[Dict]:
    detections: List[Dict] = []
    h, w = frame.shape[:2]

    scales = [0.7, 0.85, 1.0, 1.2, 1.5, 1.8]
    all_boxes: List[List[float]] = []
    all_scores: List[float] = []

    for s in scales:
        if s != 1.0:
            nh, nw = int(h * s), int(w * s)
            img = cv2.resize(frame, (nw, nh))
        else:
            img = frame
        results = model(img, conf=conf_th, iou=iou_th, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                if int(b.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                if s != 1.0:
                    x1, x2 = x1 / s, x2 / s
                    y1, y2 = y1 / s, y2 / s
                x1 = max(0, min(float(x1), w))
                y1 = max(0, min(float(y1), h))
                x2 = max(0, min(float(x2), w))
                y2 = max(0, min(float(y2), h))
                bw, bh = x2 - x1, y2 - y1
                if bw < min_size or bh < min_size:
                    continue
                ar = bw / max(bh, 1e-6)
                if not (0.2 <= ar <= 1.6):
                    continue
                area = bw * bh
                if area < 150 or area > 20000:
                    continue
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(float(b.conf[0]))

    if all_boxes:
        idxs = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf_th, iou_th)
        if len(idxs) > 0:
            for i in idxs.flatten():
                x1, y1, x2, y2 = all_boxes[i]
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(all_scores[i])
                })
    return detections


def visualize_persons(frame: np.ndarray, persons: List[Dict]) -> np.ndarray:
    img = frame.copy()
    for k, person in enumerate(persons, start=1):
        x1, y1, x2, y2 = person['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person {k} Conf:{person['confidence']:.2f}"
        tsize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - tsize[1] - 6), (x1 + tsize[0] + 6, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def run_video(video_path: str, max_frames: int = 200) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('无法打开视频文件')

    model = _init_yolo('weights/yolov8m.pt')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, total // max_frames)

    out_dir = f"outputs/persons_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    per_frame_stats = []
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if fn % skip != 0:
            continue
        persons = detect_persons_in_frame(model, frame)
        vis = visualize_persons(frame, persons)
        cv2.imwrite(os.path.join(out_dir, f"frame_{fn:06d}_persons_{len(persons)}.jpg"), vis)
        per_frame_stats.append({'frame': fn, 'persons': len(persons)})
        saved += 1
        if saved >= max_frames:
            break

    cap.release()

    report = {
        'video': video_path,
        'saved_frames': saved,
        'stats': per_frame_stats,
        'output_dir': out_dir
    }
    with open(os.path.join(out_dir, 'persons_report.json'), 'w', encoding='utf-8'):
        pass
    with open(os.path.join(out_dir, 'persons_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('用法: python person_only_detector.py <video> [max_frames]')
        raise SystemExit(0)
    video = sys.argv[1]
    maxf = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run_video(video, maxf) 