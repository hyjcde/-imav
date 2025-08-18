#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from utils.yolo_utils import safe_predict


def generate_tiles(height: int, width: int,
                   tile_size: Tuple[int, int] = (960, 960),
                   overlap_ratio: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """
    生成切片窗口坐标列表 (x1, y1, x2, y2)
    """
    tile_w, tile_h = tile_size[0], tile_size[1]
    step_w = max(1, int(tile_w * (1 - overlap_ratio)))
    step_h = max(1, int(tile_h * (1 - overlap_ratio)))

    windows: List[Tuple[int, int, int, int]] = []

    y = 0
    while True:
        if y + tile_h >= height:
            y = max(0, height - tile_h)
        x = 0
        while True:
            if x + tile_w >= width:
                x = max(0, width - tile_w)
            x2 = min(x + tile_w, width)
            y2 = min(y + tile_h, height)
            x1 = max(0, x2 - tile_w)
            y1 = max(0, y2 - tile_h)
            windows.append((x1, y1, x2, y2))
            if x + tile_w >= width:
                break
            x += step_w
        if y + tile_h >= height:
            break
        y += step_h

    # 去重
    unique = []
    seen = set()
    for w in windows:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def _generate_tiles_with_offset(height: int, width: int,
                                 tile_size: Tuple[int, int],
                                 overlap_ratio: float,
                                 offset_xy: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    tile_w, tile_h = tile_size
    step_w = max(1, int(tile_w * (1 - overlap_ratio)))
    step_h = max(1, int(tile_h * (1 - overlap_ratio)))
    start_x = max(0, min(offset_xy[0], step_w - 1))
    start_y = max(0, min(offset_xy[1], step_h - 1))

    windows: List[Tuple[int, int, int, int]] = []
    y = start_y
    while True:
        if y + tile_h >= height:
            y = max(0, height - tile_h)
        x = start_x
        while True:
            if x + tile_w >= width:
                x = max(0, width - tile_w)
            x2 = min(x + tile_w, width)
            y2 = min(y + tile_h, height)
            x1 = max(0, x2 - tile_w)
            y1 = max(0, y2 - tile_h)
            windows.append((x1, y1, x2, y2))
            if x + tile_w >= width:
                break
            x += step_w
        if y + tile_h >= height:
            break
        y += step_h

    unique = []
    seen = set()
    for w in windows:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def run_tiled_detection(model,
                        frame: np.ndarray,
                        conf_th: float = 0.20,
                        iou_th: float = 0.40,
                        tile_size: Tuple[int, int] = (960, 960),
                        overlap_ratio: float = 0.25,
                        imgsz: int = 960) -> Tuple[List[List[float]], List[float]]:
    """
    基础切片推理，返回全局坐标的框与分数列表。
    """
    h, w = frame.shape[:2]
    windows = generate_tiles(h, w, tile_size=tile_size, overlap_ratio=overlap_ratio)

    all_boxes: List[List[float]] = []
    all_scores: List[float] = []

    for (x1, y1, x2, y2) in windows:
        tile = frame[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        results = safe_predict(model, tile, conf=conf_th, iou=iou_th, imgsz=imgsz, verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls[0])
                score = float(b.conf[0])
                if cls_id != 0:
                    continue
                bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                gx1 = max(0.0, min(float(bx1) + x1, w))
                gy1 = max(0.0, min(float(by1) + y1, h))
                gx2 = max(0.0, min(float(bx2) + x1, w))
                gy2 = max(0.0, min(float(by2) + y1, h))
                if gx2 - gx1 <= 1 or gy2 - gy1 <= 1:
                    continue
                all_boxes.append([gx1, gy1, gx2, gy2])
                all_scores.append(score)

    return all_boxes, all_scores


def nms_boxes(boxes: List[List[float]], scores: List[float],
              conf_th: float, iou_th: float) -> List[int]:
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores,
                            score_threshold=conf_th, nms_threshold=iou_th)
    if isinstance(idxs, (list, tuple)):
        return [int(i) for i in idxs]
    try:
        return [int(i) for i in idxs.flatten()] if len(idxs) > 0 else []
    except Exception:
        return []


def run_tiled_detection_multi(model,
                              frame: np.ndarray,
                              conf_th: float = 0.20,
                              iou_th: float = 0.40,
                              tile_sizes: Optional[List[int]] = None,
                              overlap_ratio: float = 0.30,
                              imgsz: int = 960,
                              passes: int = 2,
                              hflip: bool = True,
                              vflip: bool = False) -> Tuple[List[List[float]], List[float]]:
    """
    高级切片推理：多尺寸、多偏移（第二遍半步对齐）、可选水平翻转 TTA。
    返回聚合后的 boxes/scores（未做最终NMS），由调用方自行 NMS。
    """
    if tile_sizes is None or len(tile_sizes) == 0:
        tile_sizes = [imgsz]

    h, w = frame.shape[:2]
    all_boxes: List[List[float]] = []
    all_scores: List[float] = []

    for ts in tile_sizes:
        tile_size = (int(ts), int(ts))
        step_w = max(1, int(tile_size[0] * (1 - overlap_ratio)))
        step_h = max(1, int(tile_size[1] * (1 - overlap_ratio)))
        offsets = [(0, 0)]
        if passes >= 2:
            offsets.append((step_w // 2, step_h // 2))

        for off in offsets:
            windows = _generate_tiles_with_offset(h, w, tile_size, overlap_ratio, off)
            for (x1, y1, x2, y2) in windows:
                tile = frame[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                # 原图推理
                results = safe_predict(model, tile, conf=conf_th, iou=iou_th, imgsz=ts, verbose=False)
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue
                    for b in boxes:
                        cls_id = int(b.cls[0])
                        score = float(b.conf[0])
                        if cls_id != 0:
                            continue
                        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                        gx1 = max(0.0, min(float(bx1) + x1, w))
                        gy1 = max(0.0, min(float(by1) + y1, h))
                        gx2 = max(0.0, min(float(bx2) + x1, w))
                        gy2 = max(0.0, min(float(by2) + y1, h))
                        if gx2 - gx1 > 1 and gy2 - gy1 > 1:
                            all_boxes.append([gx1, gy1, gx2, gy2])
                            all_scores.append(score)
                # 水平翻转 TTA
                if hflip:
                    tile_f = cv2.flip(tile, 1)
                    results_f = safe_predict(model, tile_f, conf=conf_th, iou=iou_th, imgsz=ts, verbose=False)
                    tw = tile.shape[1]
                    for r in results_f:
                        boxes = r.boxes
                        if boxes is None:
                            continue
                        for b in boxes:
                            cls_id = int(b.cls[0])
                            score = float(b.conf[0])
                            if cls_id != 0:
                                continue
                            bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                            fx1 = tw - float(bx2)
                            fx2 = tw - float(bx1)
                            gx1 = max(0.0, min(fx1 + x1, w))
                            gy1 = max(0.0, min(float(by1) + y1, h))
                            gx2 = max(0.0, min(fx2 + x1, w))
                            gy2 = max(0.0, min(float(by2) + y1, h))
                            if gx2 - gx1 > 1 and gy2 - gy1 > 1:
                                all_boxes.append([gx1, gy1, gx2, gy2])
                                all_scores.append(score)
                # 垂直翻转 TTA（可选）
                if vflip:
                    tile_fv = cv2.flip(tile, 0)
                    results_fv = safe_predict(model, tile_fv, conf=conf_th, iou=iou_th, imgsz=ts, verbose=False)
                    th = tile.shape[0]
                    for r in results_fv:
                        boxes = r.boxes
                        if boxes is None:
                            continue
                        for b in boxes:
                            cls_id = int(b.cls[0])
                            score = float(b.conf[0])
                            if cls_id != 0:
                                continue
                            bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                            fy1 = th - float(by2)
                            fy2 = th - float(by1)
                            gx1 = max(0.0, min(float(bx1) + x1, w))
                            gy1 = max(0.0, min(fy1 + y1, h))
                            gx2 = max(0.0, min(float(bx2) + x1, w))
                            gy2 = max(0.0, min(fy2 + y1, h))
                            if gx2 - gx1 > 1 and gy2 - gy1 > 1:
                                all_boxes.append([gx1, gy1, gx2, gy2])
                                all_scores.append(score)

    return all_boxes, all_scores 