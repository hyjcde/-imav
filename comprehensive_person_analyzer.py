#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合人物分析器
显示所有检测到的人物，详细记录和分析每个人与描述的匹配度
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import logging
import os
import json
from datetime import datetime
from colorama import Fore, init
import matplotlib.pyplot as plt

init(autoreset=True)
logger = logging.getLogger(__name__)

class ComprehensivePersonAnalyzer:
    """综合人物分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.detector = self._init_detector()
        
        # 检测参数
        self.min_person_size = 15
        self.confidence_threshold = 0.2  # 更低的阈值以检测更多人物
        self.nms_threshold = 0.3
        
        # 分析记录
        self.analysis_log = []
        
    def _init_detector(self):
        """初始化YOLO检测器"""
        try:
            import torch
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return original_load(*args, **kwargs)
            torch.load = patched_load
            
            try:
                model = YOLO('weights/yolov8m.pt')
                logger.info("成功加载YOLO模型: yolov8m.pt")
                return model
            finally:
                torch.load = original_load
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def detect_all_persons_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """检测帧中所有人物"""
        detections = []
        original_height, original_width = frame.shape[:2]
        
        # 多尺度检测
        scales = [0.7, 0.85, 1.0, 1.2, 1.5, 1.8]  # 更多尺度
        
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
            
            results = self.detector(scaled_frame, 
                                  conf=self.confidence_threshold,
                                  iou=self.nms_threshold,
                                  verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == 0:  # person class
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
                            
                            if self._is_valid_person(width, height, confidence):
                                all_boxes.append([x1, y1, x2, y2])
                                all_scores.append(confidence)
                                
                                person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                all_crops.append(person_crop)
        
        # NMS处理
        if all_boxes:
            indices = cv2.dnn.NMSBoxes(
                all_boxes, all_scores, 
                self.confidence_threshold, 
                self.nms_threshold
            )
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = all_boxes[i]
                    detection_info = {
                        'id': len(detections) + 1,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': all_scores[i],
                        'person_crop': all_crops[i],
                        'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                        'size': (int(x2 - x1), int(y2 - y1)),
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    detections.append(detection_info)
        
        return detections
    
    def _is_valid_person(self, width: float, height: float, confidence: float) -> bool:
        """判断是否为有效人物检测"""
        if width < self.min_person_size or height < self.min_person_size:
            return False
        
        # 放宽宽高比限制，适应更多角度
        aspect_ratio = width / height
        if not (0.2 <= aspect_ratio <= 1.2):
            return False
        
        area = width * height
        if area < 150 or area > 8000:  # 放宽面积限制
            return False
        
        return True
    
    def extract_detailed_color_features(self, person_crop: np.ndarray) -> Dict[str, Any]:
        """提取详细的颜色特征"""
        if person_crop is None or person_crop.size == 0:
            return {}
        
        height, width = person_crop.shape[:2]
        
        # 分区域分析
        regions = {
            'head': person_crop[:height//3, :],           # 头部（上1/3）
            'torso': person_crop[height//3:2*height//3, :], # 躯干（中1/3）
            'legs': person_crop[2*height//3:, :],         # 腿部（下1/3）
            'full': person_crop                           # 全身
        }
        
        color_features = {}
        
        # HSV颜色范围定义
        color_ranges = {
            'red': [[(0, 100, 80), (10, 255, 255)], [(170, 100, 80), (180, 255, 255)]],
            'orange': [[(10, 100, 80), (25, 255, 255)]],
            'yellow': [[(25, 100, 80), (35, 255, 255)]],
            'green': [[(35, 100, 80), (85, 255, 255)]],
            'blue': [[(85, 100, 80), (125, 255, 255)]],
            'purple': [[(125, 100, 80), (155, 255, 255)]],
            'pink': [[(155, 100, 80), (170, 255, 255)]],
            'white': [[(0, 0, 180), (180, 30, 255)]],
            'gray': [[(0, 0, 50), (180, 30, 180)]],
            'black': [[(0, 0, 0), (180, 255, 50)]]
        }
        
        for region_name, region_img in regions.items():
            if region_img.size == 0:
                continue
                
            hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
            total_pixels = region_img.shape[0] * region_img.shape[1]
            
            region_colors = {}
            
            for color_name, ranges in color_ranges.items():
                total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                for lower, upper in ranges:
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    total_mask = cv2.bitwise_or(total_mask, mask)
                
                color_pixels = cv2.countNonZero(total_mask)
                ratio = color_pixels / total_pixels
                
                if ratio > 0.05:  # 至少占5%
                    region_colors[color_name] = ratio
            
            color_features[region_name] = region_colors
        
        return color_features
    
    def analyze_description_match(self, person: Dict, description: str) -> Dict[str, Any]:
        """详细分析人物与描述的匹配程度"""
        # 提取颜色特征
        color_features = self.extract_detailed_color_features(person['person_crop'])
        
        # 解析描述
        description_lower = description.lower()
        
        # 颜色关键词
        color_keywords = {
            'red': ['红色', '红', 'red', 'crimson'],
            'orange': ['橙色', '橙', 'orange'],
            'yellow': ['黄色', '黄', 'yellow'],
            'green': ['绿色', '绿', 'green'],
            'blue': ['蓝色', '蓝', 'blue'],
            'purple': ['紫色', '紫', 'purple'],
            'pink': ['粉色', '粉红', 'pink'],
            'white': ['白色', '白', 'white'],
            'gray': ['灰色', '灰', 'gray', 'grey'],
            'black': ['黑色', '黑', 'black']
        }
        
        # 服装关键词
        clothing_keywords = {
            'helmet': ['头盔', '安全帽', 'helmet', 'hard hat'],
            'hat': ['帽子', '帽', 'hat', 'cap'],
            'shirt': ['上衣', '衬衫', 'shirt', 'top'],
            'jacket': ['外套', '夹克', 'jacket', 'coat'],
            'pants': ['裤子', '长裤', 'pants', 'trousers'],
            'shorts': ['短裤', 'shorts'],
            'dress': ['裙子', '连衣裙', 'dress']
        }
        
        # 解析目标特征
        target_colors = []
        target_clothing = []
        
        for color, keywords in color_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                target_colors.append(color)
        
        for clothing, keywords in clothing_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                target_clothing.append(clothing)
        
        # 计算匹配分数
        match_analysis = {
            'target_colors': target_colors,
            'target_clothing': target_clothing,
            'detected_colors': color_features,
            'color_matches': {},
            'clothing_matches': {},
            'total_score': 0.0,
            'detailed_scores': {}
        }
        
        # 颜色匹配分析
        color_score = 0.0
        if target_colors:
            for target_color in target_colors:
                best_match = 0.0
                best_region = None
                
                for region, colors in color_features.items():
                    if target_color in colors:
                        if colors[target_color] > best_match:
                            best_match = colors[target_color]
                            best_region = region
                
                if best_match > 0:
                    match_analysis['color_matches'][target_color] = {
                        'ratio': best_match,
                        'region': best_region,
                        'score': min(best_match * 2, 1.0)  # 转换为0-1分数
                    }
                    color_score += match_analysis['color_matches'][target_color]['score']
            
            color_score /= len(target_colors)
        
        # 服装匹配分析（简化版）
        clothing_score = 0.5 if target_clothing else 0.0  # 基础分数
        
        # 检测质量分数
        quality_score = person['confidence']
        
        # 综合评分
        total_score = color_score * 0.6 + clothing_score * 0.3 + quality_score * 0.1
        
        match_analysis['detailed_scores'] = {
            'color_score': color_score,
            'clothing_score': clothing_score,
            'quality_score': quality_score
        }
        match_analysis['total_score'] = total_score
        
        return match_analysis
    
    def comprehensive_analysis(self, video_path: str, description: str, max_frames: int = 20) -> Dict[str, Any]:
        """综合分析视频中的所有人物"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}🎯 综合人物分析")
        print(f"{Fore.CYAN}视频: {video_path}")
        print(f"{Fore.CYAN}目标描述: {description}")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{Fore.RED}❌ 无法打开视频文件")
            return {}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, total_frames // max_frames)
        
        output_dir = os.path.join("outputs", "comprehensive_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        all_detections = []
        frame_results = []
        frame_count = 0
        
        print(f"{Fore.YELLOW}📊 开始分析 {max_frames} 帧...")
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % skip_frames != 0:
                continue
            
            # 检测所有人物
            detections = self.detect_all_persons_in_frame(frame)
            
            if detections:
                print(f"\n{Fore.GREEN}📍 帧 {current_frame_number}: 检测到 {len(detections)} 个人物")
                
                frame_analysis = {
                    'frame_number': current_frame_number,
                    'detections': [],
                    'best_match': None,
                    'frame': frame.copy()
                }
                
                # 分析每个检测到的人物
                for person in detections:
                    match_analysis = self.analyze_description_match(person, description)
                    
                    person_analysis = {
                        'person_id': person['id'],
                        'bbox': person['bbox'],
                        'confidence': person['confidence'],
                        'size': person['size'],
                        'area': person['area'],
                        'match_analysis': match_analysis
                    }
                    
                    frame_analysis['detections'].append(person_analysis)
                    all_detections.append(person_analysis)
                    
                    # 打印详细分析
                    print(f"  👤 人物 {person['id']}:")
                    print(f"     位置: {person['bbox']}")
                    print(f"     大小: {person['size'][0]}x{person['size'][1]}")
                    print(f"     置信度: {person['confidence']:.3f}")
                    print(f"     匹配分数: {match_analysis['total_score']:.3f}")
                    
                    if match_analysis['color_matches']:
                        print(f"     颜色匹配:")
                        for color, match_info in match_analysis['color_matches'].items():
                            print(f"       {color}: {match_info['ratio']:.2%} ({match_info['region']})")
                
                # 找出最佳匹配
                if frame_analysis['detections']:
                    best_detection = max(frame_analysis['detections'], 
                                       key=lambda x: x['match_analysis']['total_score'])
                    frame_analysis['best_match'] = best_detection
                    
                    print(f"  🏆 最佳匹配: 人物 {best_detection['person_id']} (分数: {best_detection['match_analysis']['total_score']:.3f})")
                
                # 可视化并保存
                vis_frame = self.visualize_all_detections(frame, frame_analysis, description)
                output_path = f"{output_dir}/frame_{current_frame_number}_analysis.jpg"
                cv2.imwrite(output_path, vis_frame)
                
                frame_results.append(frame_analysis)
            
            frame_count += 1
        
        cap.release()
        
        # 生成综合报告
        report = self.generate_comprehensive_report(all_detections, frame_results, description, output_dir)
        
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}📋 分析完成")
        print(f"{Fore.GREEN}总计检测到: {len(all_detections)} 个人物实例")
        print(f"{Fore.GREEN}包含人物的帧数: {len(frame_results)}")
        print(f"{Fore.GREEN}结果保存在: {output_dir}")
        print(f"{Fore.CYAN}{'='*80}")
        
        return report
    
    def visualize_all_detections(self, frame: np.ndarray, frame_analysis: Dict, description: str) -> np.ndarray:
        """可视化所有检测结果"""
        vis_frame = frame.copy()
        detections = frame_analysis['detections']
        
        # 为每个人物分配不同颜色
        colors = [
            (255, 0, 0),    # 蓝色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
        ]
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            match_score = detection['match_analysis']['total_score']
            
            # 根据匹配分数选择颜色强度
            if match_score > 0.5:
                color = (0, 255, 0)  # 高分用绿色
                thickness = 4
            elif match_score > 0.3:
                color = (0, 255, 255)  # 中分用黄色
                thickness = 3
            else:
                color = colors[i % len(colors)]  # 低分用不同颜色
                thickness = 2
            
            # 绘制检测框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # 添加详细标签
            labels = [
                f"Person {detection['person_id']}",
                f"Conf: {detection['confidence']:.3f}",
                f"Match: {match_score:.3f}",
                f"Size: {detection['size'][0]}x{detection['size'][1]}"
            ]
            
            # 绘制标签背景
            label_height = len(labels) * 20 + 10
            cv2.rectangle(vis_frame, (x1, y1 - label_height), (x1 + 200, y1), color, -1)
            
            # 绘制文字
            for j, label in enumerate(labels):
                y_pos = y1 - label_height + 15 + j * 20
                cv2.putText(vis_frame, label, (x1 + 5, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 显示颜色匹配信息
            if detection['match_analysis']['color_matches']:
                y_offset = y2 + 20
                for color_name, match_info in detection['match_analysis']['color_matches'].items():
                    color_text = f"{color_name}: {match_info['ratio']:.1%} ({match_info['region']})"
                    cv2.putText(vis_frame, color_text, (x1, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 15
        
        # 添加帧信息
        frame_info = [
            f"Frame: {frame_analysis['frame_number']}",
            f"Persons: {len(detections)}",
            f"Target: {description}",
            f"Best Match: {frame_analysis['best_match']['match_analysis']['total_score']:.3f}" if frame_analysis['best_match'] else "No Match"
        ]
        
        for i, info in enumerate(frame_info):
            cv2.putText(vis_frame, info, (10, 30 + i * 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def generate_comprehensive_report(self, all_detections: List[Dict], frame_results: List[Dict], 
                                    description: str, output_dir: str) -> Dict[str, Any]:
        """生成综合分析报告"""
        # 按匹配分数排序
        sorted_detections = sorted(all_detections, key=lambda x: x['match_analysis']['total_score'], reverse=True)
        
        # 统计信息
        total_detections = len(all_detections)
        total_frames = len(frame_results)
        avg_score = np.mean([d['match_analysis']['total_score'] for d in all_detections])
        
        # 最佳匹配分析
        best_matches = sorted_detections[:10]  # 前10个最佳匹配
        
        # 生成JSON报告
        report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'description': description,
                'total_detections': total_detections,
                'total_frames': total_frames,
                'average_score': float(avg_score)
            },
            'best_matches': [],
            'frame_summary': []
        }
        
        # 最佳匹配详情
        for i, detection in enumerate(best_matches):
            match_info = {
                'rank': i + 1,
                'person_id': detection['person_id'],
                'total_score': detection['match_analysis']['total_score'],
                'bbox': detection['bbox'],
                'size': detection['size'],
                'confidence': detection['confidence'],
                'color_matches': detection['match_analysis']['color_matches'],
                'detailed_scores': detection['match_analysis']['detailed_scores']
            }
            report['best_matches'].append(match_info)
        
        # 帧汇总
        for frame_result in frame_results:
            frame_summary = {
                'frame_number': frame_result['frame_number'],
                'person_count': len(frame_result['detections']),
                'best_score': frame_result['best_match']['match_analysis']['total_score'] if frame_result['best_match'] else 0.0
            }
            report['frame_summary'].append(frame_summary)
        
        # 保存JSON报告
        with open(f"{output_dir}/analysis_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可视化报告
        self.create_visual_report(report, output_dir)
        
        # 打印最佳匹配结果
        print(f"\n{Fore.CYAN}🏆 前10个最佳匹配:")
        for match in best_matches:
            print(f"  {match['match_analysis']['total_score']:.3f} - 人物ID {match['person_id']} "
                  f"(大小: {match['size'][0]}x{match['size'][1]})")
            if match['match_analysis']['color_matches']:
                for color, info in match['match_analysis']['color_matches'].items():
                    print(f"    {color}: {info['ratio']:.1%} 在 {info['region']}")
        
        return report
    
    def create_visual_report(self, report: Dict, output_dir: str):
        """创建可视化报告"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'人物检测分析报告 - {report["analysis_info"]["description"]}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 匹配分数分布
        scores = [m['total_score'] for m in report['best_matches']]
        ax1.bar(range(len(scores)), scores, alpha=0.7, color='skyblue')
        ax1.set_title('最佳匹配分数分布')
        ax1.set_xlabel('排名')
        ax1.set_ylabel('匹配分数')
        ax1.set_xticks(range(len(scores)))
        ax1.set_xticklabels([f'#{i+1}' for i in range(len(scores))])
        
        # 2. 每帧人物数量
        frame_nums = [f['frame_number'] for f in report['frame_summary']]
        person_counts = [f['person_count'] for f in report['frame_summary']]
        ax2.plot(frame_nums, person_counts, 'o-', color='orange')
        ax2.set_title('每帧检测人物数量')
        ax2.set_xlabel('帧号')
        ax2.set_ylabel('人物数量')
        ax2.grid(True, alpha=0.3)
        
        # 3. 人物大小分布
        sizes = [m['size'][0] * m['size'][1] for m in report['best_matches']]
        ax3.hist(sizes, bins=10, alpha=0.7, color='lightgreen')
        ax3.set_title('人物大小分布 (像素²)')
        ax3.set_xlabel('面积 (像素²)')
        ax3.set_ylabel('数量')
        
        # 4. 置信度vs匹配分数
        confidences = [m['confidence'] for m in report['best_matches']]
        match_scores = [m['total_score'] for m in report['best_matches']]
        ax4.scatter(confidences, match_scores, alpha=0.7, color='red')
        ax4.set_title('检测置信度 vs 匹配分数')
        ax4.set_xlabel('检测置信度')
        ax4.set_ylabel('匹配分数')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visual_report.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"{Fore.GREEN}📊 可视化报告已保存: visual_report.png")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 3:
        video_path = "data/DJI_20250807124730_0002_S.MP4"
        description = "a man with red helmet"
        max_frames = 15
        print(f"{Fore.CYAN}使用默认参数: {video_path}, '{description}'")
    else:
        video_path = sys.argv[1]
        description = sys.argv[2]
        max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    try:
        analyzer = ComprehensivePersonAnalyzer()
        report = analyzer.comprehensive_analysis(video_path, description, max_frames)
        
    except Exception as e:
        print(f"{Fore.RED}❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 