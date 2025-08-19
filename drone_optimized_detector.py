#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机影像优化的人物检测系统
针对俯视角度、远距离、小目标等无人机拍摄特点进行优化
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import logging
from text_matcher import TextMatcher
import os
import matplotlib.pyplot as plt
from colorama import Fore, init
from utils.tiling import run_tiled_detection, nms_boxes
from utils.tiling import run_tiled_detection_multi
from utils.yolo_utils import safe_predict
from utils.clip_utils import load_clip, compute_clip_similarity

init(autoreset=True)
logger = logging.getLogger(__name__)

class DroneOptimizedPersonFinder:
    """针对无人机影像优化的人物查找器"""
    
    def __init__(self,
                 model_name: str = 'weights/yolov8s.pt',
                 min_person_size: int = 20,
                 confidence_threshold: float = 0.30,
                 nms_threshold: float = 0.40,
                 tile_size: int = 832,
                 tile_overlap: float = 0.30,
                 tile_imgsz: int = 832):
        """初始化优化检测器"""
        self.model_name = model_name
        self.detector = self._init_detector()
        self.matcher = TextMatcher()
        
        # 无人机检测优化参数
        self.min_person_size = int(min_person_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        
        # 切片推理参数
        self.tile_size = (int(tile_size), int(tile_size))
        self.tile_overlap = float(tile_overlap)
        self.tile_imgsz = int(tile_imgsz)
        
    def _init_detector(self):
        """初始化YOLO检测器"""
        try:
            import torch
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            # 处理PyTorch权重加载
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return original_load(*args, **kwargs)
            torch.load = patched_load
            
            try:
                # 使用传入的模型权重（默认 yolov8s）
                weights_path = getattr(self, 'model_name', 'weights/yolov8s.pt')
                model = YOLO(weights_path)
                logger.info(f"成功加载YOLO模型: {os.path.basename(weights_path)}")
                return model
            finally:
                torch.load = original_load
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def detect_all_persons_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        优化的人物检测，专门针对无人机影像
        """
        detections = []
        original_height, original_width = frame.shape[:2]
        # 有效最小人物尺寸：像素阈值与比例阈值取更大者
        ratio = float(getattr(self, 'min_size_ratio', 0.0))
        eff_min_size = max(self.min_person_size, int(min(original_height, original_width) * ratio))

        # 首选：切片推理，提升小目标召回
        if getattr(self, 'use_tta', False):
            tiled_boxes, tiled_scores = run_tiled_detection_multi(
                self.detector,
                frame,
                conf_th=self.confidence_threshold,
                iou_th=self.nms_threshold,
                tile_sizes=getattr(self, 'tile_sizes_list', [self.tile_imgsz, max(640, self.tile_imgsz - 128)]),
                overlap_ratio=max(self.tile_overlap, 0.3),
                imgsz=self.tile_imgsz,
                passes=2,
                hflip=bool(getattr(self, 'tta_hflip', True)),
                vflip=bool(getattr(self, 'tta_vflip', False))
            )
        else:
            tiled_boxes, tiled_scores = run_tiled_detection(
                self.detector,
                frame,
                conf_th=self.confidence_threshold,
                iou_th=self.nms_threshold,
                tile_size=self.tile_size,
                overlap_ratio=self.tile_overlap,
                imgsz=self.tile_imgsz
            )

        keep_idx = nms_boxes(tiled_boxes, tiled_scores, self.confidence_threshold, self.nms_threshold)
        for i in keep_idx:
            x1, y1, x2, y2 = tiled_boxes[i]
            confidence = float(tiled_scores[i])
            if (x2 - x1) >= eff_min_size and (y2 - y1) >= eff_min_size:
                person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'person_crop': person_crop,
                    'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                    'size': (int(x2 - x1), int(y2 - y1))
                })

        # 渐进式补充：若开启 progressive，则逐步降低 conf/提高 overlap，并并集合并
        if getattr(self, 'use_progressive', False):
            tiers = [
                # (conf, iou, overlap, imgsz)
                (max(0.10, self.confidence_threshold - 0.05), self.nms_threshold, min(0.55, self.tile_overlap + 0.10), max(self.tile_imgsz, 1024)),
                (max(0.06, self.confidence_threshold - 0.09), self.nms_threshold, min(0.60, self.tile_overlap + 0.15), max(self.tile_imgsz, 1152))
            ]
            base_boxes = [tiled_boxes[k] for k in keep_idx]
            base_scores = [tiled_scores[k] for k in keep_idx]
            for (t_conf, t_iou, t_overlap, t_imgsz) in tiers:
                if getattr(self, 'use_tta', False):
                    add_boxes, add_scores = run_tiled_detection_multi(
                        self.detector, frame, conf_th=t_conf, iou_th=t_iou,
                        tile_sizes=getattr(self, 'tile_sizes_list', [t_imgsz]),
                        overlap_ratio=t_overlap, imgsz=t_imgsz, passes=2,
                        hflip=bool(getattr(self, 'tta_hflip', True)),
                        vflip=bool(getattr(self, 'tta_vflip', False))
                    )
                else:
                    add_boxes, add_scores = run_tiled_detection(
                        self.detector, frame, conf_th=t_conf, iou_th=t_iou,
                        tile_size=(t_imgsz, t_imgsz), overlap_ratio=t_overlap, imgsz=t_imgsz
                    )
                base_boxes.extend(add_boxes)
                base_scores.extend(add_scores)
                # 全局NMS
                keep = nms_boxes(base_boxes, base_scores, t_conf, t_iou)
                # 重建 detections
                detections = []
                for idx in keep:
                    x1, y1, x2, y2 = base_boxes[idx]
                    sc = float(base_scores[idx])
                    if (x2 - x1) >= eff_min_size and (y2 - y1) >= eff_min_size:
                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': sc,
                            'person_crop': crop,
                            'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                            'size': (int(x2 - x1), int(y2 - y1))
                        })
        if detections:
            return detections

        # 回退：原多尺度
        detections = []
        original_height, original_width = frame.shape[:2]
        scales = [1.0, 1.2, 1.5]
        all_boxes = []
        all_scores = []
        all_crops = []
        for scale in scales:
            if scale != 1.0:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                scaled_frame = cv2.resize(frame, (new_width, new_height))
            else:
                scaled_frame = frame.copy()
            results = safe_predict(self.detector, scaled_frame, conf=self.confidence_threshold, iou=self.nms_threshold)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        if class_id == 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            if scale != 1.0:
                                x1, x2 = x1 / scale, x2 / scale
                                y1, y2 = y1 / scale, y2 / scale
                            x1 = max(0, min(x1, original_width))
                            y1 = max(0, min(y1, original_height))
                            x2 = max(0, min(x2, original_width))
                            y2 = max(0, min(y2, original_height))
                            width = x2 - x1
                            height = y2 - y1
                            if width >= eff_min_size and height >= eff_min_size:
                                all_boxes.append([x1, y1, x2, y2])
                                all_scores.append(confidence)
                                person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                all_crops.append(person_crop)
        if all_boxes:
            indices = cv2.dnn.NMSBoxes(all_boxes, all_scores,
                self.confidence_threshold, 
                                       self.nms_threshold)
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = all_boxes[i]
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': all_scores[i],
                        'person_crop': all_crops[i],
                        'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                        'size': (int(x2 - x1), int(y2 - y1))
                    })
        return detections
    
    def find_all_persons_in_video(self, video_path: str, max_frames: int = 50) -> Dict[str, Any]:
        """
        在视频中找到所有人物，不进行文本匹配
        """
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}开始检测视频中的所有人物")
        print(f"{Fore.CYAN}视频文件: {video_path}")
        print(f"{Fore.CYAN}{'='*60}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return {"status": "error", "message": "无法打开视频文件"}
        
        frame_detections = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"总帧数: {total_frames}, 计划处理: {min(max_frames, total_frames)} 帧")
        
        # 计算跳帧间隔
        skip_frames = max(1, total_frames // max_frames) if max_frames < total_frames else 1
        
        all_persons = []
        frame_info = []
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % skip_frames != 0:
                continue
            
            # 检测当前帧中的所有人物
            detections = self.detect_all_persons_in_frame(frame)
            
            if detections:
                frame_info.append({
                    'frame_number': current_frame_number,
                    'detections': detections,
                    'frame': frame.copy()
                })
                
                print(f"{Fore.GREEN}帧 {current_frame_number}: 检测到 {len(detections)} 个人物")
                
                # 记录每个人物的详细信息
                for i, detection in enumerate(detections):
                    person_info = {
                        'frame_number': current_frame_number,
                        'person_id': f"frame_{current_frame_number}_person_{i}",
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'size': detection['size'],
                        'person_crop': detection['person_crop']
                    }
                    all_persons.append(person_info)
            
            frame_count += 1
            
            # 显示进度
            if frame_count % 10 == 0:
                progress = (frame_count / max_frames) * 100
                print(f"{Fore.YELLOW}处理进度: {progress:.1f}%")
        
        cap.release()
        
        print(f"\n{Fore.GREEN}检测完成！")
        print(f"{Fore.GREEN}总计找到 {len(all_persons)} 个人物实例")
        print(f"{Fore.GREEN}分布在 {len(frame_info)} 帧中")
        
        return {
            "status": "success",
            "total_persons": len(all_persons),
            "total_frames_with_persons": len(frame_info),
            "persons": all_persons,
            "frame_info": frame_info
        }
    
    def save_all_detections(self, results: Dict[str, Any], output_dir: str = "outputs/all_detections"):
        """
        保存所有检测到的人物
        """
        if results["status"] != "success":
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{Fore.YELLOW}正在保存所有检测结果...")
        
        persons = results["persons"]
        frame_info = results["frame_info"]
        
        # 先保存每帧的“全体人员合成图”
        for finfo in frame_info:
            frame_number = finfo['frame_number']
            frame = finfo['frame']
            dets = finfo['detections']
            if frame is None or not dets:
                continue
            canvas = frame.copy()
            for k, det in enumerate(dets, start=1):
                x1, y1, x2, y2 = det['bbox']
                conf = det.get('confidence', 0.0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"P{k} {conf:.2f}"
                cv2.putText(canvas, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_number}_all_persons.jpg"), canvas)
        
        # 再保存逐人可视化小图
        for i, person in enumerate(persons):
            frame_number = person['frame_number']
            bbox = person['bbox']
            confidence = person['confidence']
            size = person['size']
            
            frame = None
            for finfo in frame_info:
                if finfo['frame_number'] == frame_number:
                    frame = finfo['frame']
                    break
            
            if frame is not None:
                display_frame = frame.copy()
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {i+1}\nConf: {confidence:.3f}\nSize: {size[0]}x{size[1]}"
                label_lines = label.split('\n')
                for j, line in enumerate(label_lines):
                    label_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(display_frame, 
                                (x1, y1 - 20 - j*15), 
                                (x1 + label_size[0], y1 - 5 - j*15), 
                                (0, 255, 0), -1)
                    cv2.putText(display_frame, line, 
                              (x1, y1 - 10 - j*15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                output_filename = f"person_{i+1:03d}_frame_{frame_number}_conf_{confidence:.3f}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, display_frame)
        
        # 创建统计摘要
        self._create_detection_summary(results, output_dir)
        
        print(f"{Fore.GREEN}所有检测结果已保存到: {output_dir}")
        print(f"{Fore.GREEN}共保存 {len(persons)} 个人物检测结果")
    
    def _create_detection_summary(self, results: Dict[str, Any], output_dir: str):
        """创建检测结果统计摘要"""
        persons = results["persons"]
        
        # 统计信息
        total_persons = len(persons)
        frames_with_persons = len(set([p['frame_number'] for p in persons]))
        avg_confidence = np.mean([p['confidence'] for p in persons])
        avg_size = np.mean([p['size'][0] * p['size'][1] for p in persons])
        
        # 按置信度排序
        sorted_persons = sorted(persons, key=lambda x: x['confidence'], reverse=True)
        
        # 创建可视化摘要
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('无人机人物检测统计摘要', fontsize=16, fontweight='bold')
        
        # 1. 置信度分布
        confidences = [p['confidence'] for p in persons]
        ax1.hist(confidences, bins=20, alpha=0.7, color='blue')
        ax1.set_title('置信度分布')
        ax1.set_xlabel('置信度')
        ax1.set_ylabel('数量')
        ax1.axvline(avg_confidence, color='red', linestyle='--', label=f'平均值: {avg_confidence:.3f}')
        ax1.legend()
        
        # 2. 人物大小分布
        sizes = [p['size'][0] * p['size'][1] for p in persons]
        ax2.hist(sizes, bins=20, alpha=0.7, color='green')
        ax2.set_title('人物大小分布 (像素²)')
        ax2.set_xlabel('面积 (像素²)')
        ax2.set_ylabel('数量')
        ax2.axvline(avg_size, color='red', linestyle='--', label=f'平均值: {avg_size:.0f}')
        ax2.legend()
        
        # 3. 每帧人物数量
        frame_counts = {}
        for p in persons:
            frame_num = p['frame_number']
            frame_counts[frame_num] = frame_counts.get(frame_num, 0) + 1
        
        frames = list(frame_counts.keys())
        counts = list(frame_counts.values())
        ax3.bar(range(len(frames)), counts, alpha=0.7, color='orange')
        ax3.set_title('每帧检测到的人物数量')
        ax3.set_xlabel('帧序号')
        ax3.set_ylabel('人物数量')
        if len(frames) <= 20:
            ax3.set_xticks(range(len(frames)))
            ax3.set_xticklabels([f"{f}" for f in frames], rotation=45)
        
        # 4. 显示最佳检测结果
        if total_persons > 0:
            best_person = sorted_persons[0]
            if best_person['person_crop'].size > 0:
                person_crop_rgb = cv2.cvtColor(best_person['person_crop'], cv2.COLOR_BGR2RGB)
                ax4.imshow(person_crop_rgb)
                ax4.set_title(f'最佳检测结果\n帧 {best_person["frame_number"]}, 置信度 {best_person["confidence"]:.3f}')
                ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存摘要图
        summary_path = os.path.join(output_dir, "detection_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存文本摘要
        summary_text_path = os.path.join(output_dir, "detection_summary.txt")
        with open(summary_text_path, 'w', encoding='utf-8') as f:
            f.write("无人机人物检测统计摘要\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"总检测人物数: {total_persons}\n")
            f.write(f"包含人物的帧数: {frames_with_persons}\n")
            f.write(f"平均置信度: {avg_confidence:.3f}\n")
            f.write(f"平均人物大小: {avg_size:.0f} 像素²\n\n")
            
            f.write("前10个最佳检测结果:\n")
            f.write("-" * 40 + "\n")
            for i, person in enumerate(sorted_persons[:10]):
                f.write(f"{i+1:2d}. 帧 {person['frame_number']:4d}, "
                       f"置信度 {person['confidence']:.3f}, "
                       f"大小 {person['size'][0]}x{person['size'][1]}\n")
        
        print(f"  保存统计摘要: detection_summary.png")
        print(f"  保存文本摘要: detection_summary.txt")

def main():
    """主函数 - 找到所有人物"""
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='UAV optimized person finder with tiled inference')
    parser.add_argument('video', nargs='?', default='data/DJI_20250807124730_0002_S.MP4', help='video path')
    parser.add_argument('max_frames', nargs='?', type=int, default=30, help='max frames to process')
    parser.add_argument('--conf', type=float, default=0.30, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.40, help='NMS IoU threshold')
    parser.add_argument('--min-size', type=int, default=20, help='minimum person bbox size (pixels)')
    parser.add_argument('--weights', type=str, default='weights/yolov8s.pt', help='YOLO weights path')
    parser.add_argument('--tile', type=int, default=832, help='tile size (square)')
    parser.add_argument('--overlap', type=float, default=0.30, help='tile overlap ratio')
    parser.add_argument('--imgsz', type=int, default=832, help='YOLO input size for tiles')
    parser.add_argument('--tta', action='store_true', help='enable multi-size+offset+flip TTA for higher recall')
    parser.add_argument('--tile-sizes', type=str, default='', help='comma-separated tile sizes for TTA, e.g., 640,832,960')
    parser.add_argument('--vflip', action='store_true', help='also enable vertical flip during TTA')
    parser.add_argument('--min-size-ratio', type=float, default=0.0, help='min size as ratio of min(frame_w, frame_h)')
    parser.add_argument('--progressive', action='store_true', help='enable progressive low-conf passes to boost recall')
    parser.add_argument('--text', type=str, default='', help='optional text description to CLIP-rerank persons, e.g., "red helmet"')
    parser.add_argument('--clip-w', type=float, default=0.5, help='weight of CLIP score in fusion [0-1]')
    args = parser.parse_args()

    video_path = args.video
    max_frames = args.max_frames

    try:
        finder = DroneOptimizedPersonFinder(
            model_name=args.weights,
            min_person_size=args.min_size,
            confidence_threshold=args.conf,
            nms_threshold=args.iou,
            tile_size=args.tile,
            tile_overlap=args.overlap,
            tile_imgsz=args.imgsz
        )
        # toggle TTA
        setattr(finder, 'use_tta', bool(args.tta))
        # parse tile sizes list for TTA
        if args.tile_sizes.strip():
            try:
                sizes = [int(x) for x in args.tile_sizes.split(',') if x.strip()]
                if sizes:
                    setattr(finder, 'tile_sizes_list', sizes)
            except Exception:
                pass
        setattr(finder, 'tta_hflip', True)
        setattr(finder, 'tta_vflip', bool(args.vflip))
        setattr(finder, 'min_size_ratio', float(args.min_size_ratio))
        setattr(finder, 'use_progressive', bool(args.progressive))
        results = finder.find_all_persons_in_video(video_path, max_frames)
        if results["status"] == "success":
            # If text provided, run CLIP rerank/fusion then save
            if args.text:
                clip_ctx = load_clip()
                if clip_ctx is not None:
                    # For each detection attach clip score and reorder
                    for finfo in results['frame_info']:
                        for det in finfo['detections']:
                            crop = det.get('person_crop')
                            s = compute_clip_similarity(clip_ctx, crop, args.text)
                            det['clip_score'] = float(s) if s is not None else 0.0
                    # Reorder persons list by fused score
                    fused = []
                    for finfo in results['frame_info']:
                        for det in finfo['detections']:
                            base = float(det.get('confidence', 0.0))
                            clip_s = float(det.get('clip_score', 0.0))
                            fused_score = (1.0 - args.clip_w) * base + args.clip_w * clip_s
                            fused.append((finfo['frame_number'], fused_score, det))
                    fused.sort(key=lambda x: x[1], reverse=True)
                    # Replace flat persons list ordered by fused score
                    results['persons'] = [
                        {
                            'frame_number': fn,
                            **det,
                            'fused_score': sc
                        } for (fn, sc, det) in fused
                    ]
                # else: silently fallback
            finder.save_all_detections(results)
    except Exception as e:
        print(f"{Fore.RED}错误: {e}")

if __name__ == "__main__":
    main() 