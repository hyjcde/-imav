#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用人物查找系统
支持任意文本描述，提供高质量可视化和结果保存
基于YOLOv8检测和智能文本匹配
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any, Optional
import logging
import os
import json
from datetime import datetime
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import re
from utils.tiling import run_tiled_detection, nms_boxes
from utils.yolo_utils import safe_predict

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

init(autoreset=True)
logger = logging.getLogger(__name__)

class UniversalPersonFinder:
    """通用人物查找系统"""
    
    def __init__(self):
        """初始化系统"""
        self.detector = self._init_detector()
        
        # 检测参数（更低阈值+更小尺寸以捕捉更多行人）
        self.min_person_size = 10
        self.confidence_threshold = 0.20
        self.nms_threshold = 0.4
        
        # 扩展颜色词典
        self.color_keywords = {
            'red': {
                'cn': ['红色', '红', '赤色', '朱红', '大红', '深红', '浅红'],
                'en': ['red', 'crimson', 'scarlet', 'burgundy', 'maroon']
            },
            'blue': {
                'cn': ['蓝色', '蓝', '深蓝', '浅蓝', '天蓝', '海蓝'],
                'en': ['blue', 'navy', 'cyan', 'azure', 'cobalt']
            },
            'green': {
                'cn': ['绿色', '绿', '深绿', '浅绿', '翠绿', '草绿'],
                'en': ['green', 'lime', 'forest', 'olive', 'emerald']
            },
            'yellow': {
                'cn': ['黄色', '黄', '金黄', '淡黄', '亮黄'],
                'en': ['yellow', 'gold', 'amber', 'lemon']
            },
            'orange': {
                'cn': ['橙色', '橙', '桔色', '橘色'],
                'en': ['orange', 'tangerine', 'peach']
            },
            'purple': {
                'cn': ['紫色', '紫', '深紫', '浅紫'],
                'en': ['purple', 'violet', 'lavender', 'magenta']
            },
            'white': {
                'cn': ['白色', '白', '纯白', '米白'],
                'en': ['white', 'ivory', 'cream']
            },
            'black': {
                'cn': ['黑色', '黑', '深黑'],
                'en': ['black', 'dark']
            },
            'gray': {
                'cn': ['灰色', '灰', '深灰', '浅灰'],
                'en': ['gray', 'grey', 'silver']
            },
            'brown': {
                'cn': ['棕色', '褐色', '咖啡色', '土黄'],
                'en': ['brown', 'tan', 'beige', 'khaki']
            },
            'pink': {
                'cn': ['粉色', '粉红', '樱花色'],
                'en': ['pink', 'rose']
            }
        }
        
        # 扩展服装词典
        self.clothing_keywords = {
            'helmet': {
                'cn': ['头盔', '安全帽', '工地帽', '防护帽'],
                'en': ['helmet', 'hard hat', 'safety helmet']
            },
            'hat': {
                'cn': ['帽子', '帽', '棒球帽', '遮阳帽'],
                'en': ['hat', 'cap', 'baseball cap']
            },
            'shirt': {
                'cn': ['上衣', '衬衫', '体恤', 'T恤', '短袖'],
                'en': ['shirt', 't-shirt', 'top', 'blouse']
            },
            'jacket': {
                'cn': ['外套', '夹克', '风衣', '大衣'],
                'en': ['jacket', 'coat', 'windbreaker']
            },
            'vest': {
                'cn': ['背心', '马甲', '反光背心'],
                'en': ['vest', 'waistcoat', 'safety vest']
            },
            'pants': {
                'cn': ['裤子', '长裤', '工装裤'],
                'en': ['pants', 'trousers', 'jeans']
            },
            'shorts': {
                'cn': ['短裤'],
                'en': ['shorts']
            },
            'uniform': {
                'cn': ['制服', '工装', '工作服'],
                'en': ['uniform', 'workwear']
            }
        }
        
        # 位置和动作词典
        self.position_keywords = {
            'standing': {
                'cn': ['站着', '站立', '直立'],
                'en': ['standing', 'upright']
            },
            'walking': {
                'cn': ['走路', '行走', '步行'],
                'en': ['walking', 'moving']
            },
            'sitting': {
                'cn': ['坐着', '坐下'],
                'en': ['sitting', 'seated']
            },
            'working': {
                'cn': ['工作', '劳动', '施工'],
                'en': ['working', 'laboring']
            },
            'center': {
                'cn': ['中间', '中央', '正中'],
                'en': ['center', 'middle', 'central']
            },
            'left': {
                'cn': ['左边', '左侧'],
                'en': ['left', 'left side']
            },
            'right': {
                'cn': ['右边', '右侧'],
                'en': ['right', 'right side']
            },
            'alone': {
                'cn': ['单独', '一个人', '独自'],
                'en': ['alone', 'single', 'individual']
            },
            'group': {
                'cn': ['群体', '多人', '一群'],
                'en': ['group', 'multiple', 'crowd']
            }
        }
        
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
    
    def detect_persons_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """检测帧中的所有人物"""
        detections = []
        original_height, original_width = frame.shape[:2]

        # 先进行切片推理，提升小目标召回（航拍友好）
        tiled_boxes, tiled_scores = run_tiled_detection(
            self.detector,
            frame,
            conf_th=self.confidence_threshold,
            iou_th=self.nms_threshold,
            tile_size=(960, 960),
            overlap_ratio=0.3,
            imgsz=960
        )
        keep_idx = nms_boxes(tiled_boxes, tiled_scores, self.confidence_threshold, self.nms_threshold)
        for i in keep_idx:
            x1, y1, x2, y2 = tiled_boxes[i]
            confidence = float(tiled_scores[i])
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            detections.append({
                'id': len(detections) + 1,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': confidence,
                'person_crop': person_crop,
                'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                'size': (int(x2 - x1), int(y2 - y1)),
                'area': int((x2 - x1) * (y2 - y1))
            })

        if detections:
            return detections

        # 回退：原多尺度（保证兼容）
        scales = [0.8, 1.0, 1.3, 1.6]
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
            results = safe_predict(self.detector, scaled_frame,
                                  conf=self.confidence_threshold,
                                  iou=self.nms_threshold,
                                  verbose=False)
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
                            if self._is_valid_person(width, height, confidence):
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
                        'id': len(detections) + 1,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': all_scores[i],
                        'person_crop': all_crops[i],
                        'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                        'size': (int(x2 - x1), int(y2 - y1)),
                        'area': int((x2 - x1) * (y2 - y1))
                    })
        return detections
    
    def _is_valid_person(self, width: float, height: float, confidence: float) -> bool:
        """判断是否为有效人物检测"""
        if width < self.min_person_size or height < self.min_person_size:
            return False
        
        aspect_ratio = width / height
        if not (0.2 <= aspect_ratio <= 1.5):
            return False
        
        area = width * height
        if area < 200 or area > 12000:
            return False
        
        return True
    
    def extract_color_features(self, person_crop: np.ndarray) -> Dict[str, Dict[str, float]]:
        """提取颜色特征"""
        if person_crop is None or person_crop.size == 0:
            return {}
        
        height, width = person_crop.shape[:2]
        
        # 分区域分析（头部、躯干、腿部）
        regions = {
            'head': person_crop[:height//3, :],           
            'torso': person_crop[height//3:2*height//3, :], 
            'legs': person_crop[2*height//3:, :],         
            'full': person_crop                           
        }
        
        # HSV颜色范围
        color_ranges = {
            'red': [[(0, 120, 70), (10, 255, 255)], [(170, 120, 70), (180, 255, 255)]],
            'blue': [[(100, 120, 70), (130, 255, 255)]],
            'green': [[(40, 120, 70), (80, 255, 255)]],
            'yellow': [[(20, 120, 70), (30, 255, 255)]],
            'orange': [[(10, 120, 70), (20, 255, 255)]],
            'purple': [[(130, 120, 70), (160, 255, 255)]],
            'white': [[(0, 0, 200), (180, 30, 255)]],
            'black': [[(0, 0, 0), (180, 255, 70)]],
            'gray': [[(0, 0, 70), (180, 30, 200)]],
            'brown': [[(10, 50, 20), (20, 255, 200)]],
            'pink': [[(160, 120, 70), (170, 255, 255)]]
        }
        
        color_features = {}
        
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
    
    def parse_text_description(self, description: str) -> Dict[str, List[str]]:
        """解析文本描述，提取关键特征"""
        description_lower = description.lower()
        
        parsed_features = {
            'colors': [],
            'clothing': [],
            'positions': [],
            'raw_text': description
        }
        
        # 提取颜色
        for color, keywords in self.color_keywords.items():
            all_keywords = keywords['cn'] + keywords['en']
            if any(keyword in description_lower for keyword in all_keywords):
                parsed_features['colors'].append(color)
        
        # 提取服装
        for clothing, keywords in self.clothing_keywords.items():
            all_keywords = keywords['cn'] + keywords['en']
            if any(keyword in description_lower for keyword in all_keywords):
                parsed_features['clothing'].append(clothing)
        
        # 提取位置/动作
        for position, keywords in self.position_keywords.items():
            all_keywords = keywords['cn'] + keywords['en']
            if any(keyword in description_lower for keyword in all_keywords):
                parsed_features['positions'].append(position)
        
        return parsed_features
    
    def calculate_match_score(self, person: Dict, description_features: Dict) -> Dict[str, Any]:
        """计算匹配分数"""
        color_features = self.extract_color_features(person['person_crop'])
        
        match_analysis = {
            'color_score': 0.0,
            'clothing_score': 0.0,
            'position_score': 0.0,
            'quality_score': person['confidence'],
            'total_score': 0.0,
            'matched_features': {
                'colors': {},
                'clothing': [],
                'positions': []
            }
        }
        
        # 颜色匹配分析
        if description_features['colors']:
            color_matches = 0
            total_color_score = 0.0
            
            for target_color in description_features['colors']:
                best_match = 0.0
                best_region = None
                
                for region, colors in color_features.items():
                    if target_color in colors:
                        if colors[target_color] > best_match:
                            best_match = colors[target_color]
                            best_region = region
                
                if best_match > 0.05:  # 至少5%的颜色匹配
                    color_matches += 1
                    score = min(best_match * 2, 1.0)
                    total_color_score += score
                    match_analysis['matched_features']['colors'][target_color] = {
                        'ratio': best_match,
                        'region': best_region,
                        'score': score
                    }
            
            if color_matches > 0:
                match_analysis['color_score'] = total_color_score / len(description_features['colors'])
        
        # 服装匹配（基础分数）
        if description_features['clothing']:
            match_analysis['clothing_score'] = 0.6  # 基础服装分数
            match_analysis['matched_features']['clothing'] = description_features['clothing']
        
        # 位置匹配（简化）
        if description_features['positions']:
            match_analysis['position_score'] = 0.5  # 基础位置分数
            match_analysis['matched_features']['positions'] = description_features['positions']
        
        # 综合评分
        weights = {
            'color': 0.5 if description_features['colors'] else 0.0,
            'clothing': 0.3 if description_features['clothing'] else 0.0,
            'position': 0.1 if description_features['positions'] else 0.0,
            'quality': 0.1
        }
        
        # 重新分配权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        
        match_analysis['total_score'] = (
            match_analysis['color_score'] * weights['color'] +
            match_analysis['clothing_score'] * weights['clothing'] +
            match_analysis['position_score'] * weights['position'] +
            match_analysis['quality_score'] * weights['quality']
        )
        
        return match_analysis
    
    def find_best_matches(self, video_path: str, description: str, max_frames: int = 20) -> Dict[str, Any]:
        """查找最佳匹配"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}🔍 通用人物查找系统")
        print(f"{Fore.CYAN}视频: {video_path}")
        print(f"{Fore.CYAN}目标描述: {Style.BRIGHT}{description}")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        # 解析描述
        description_features = self.parse_text_description(description)
        print(f"{Fore.YELLOW}📝 解析的特征:")
        print(f"   颜色: {description_features['colors']}")
        print(f"   服装: {description_features['clothing']}")
        print(f"   位置/动作: {description_features['positions']}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{Fore.RED}❌ 无法打开视频文件")
            return {}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, total_frames // max_frames)
        
        output_dir = os.path.join("outputs", f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        
        all_matches = []
        frame_count = 0
        
        print(f"{Fore.YELLOW}🔍 开始搜索 {max_frames} 帧...")
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % skip_frames != 0:
                continue
            
            # 检测人物
            detections = self.detect_persons_in_frame(frame)
            
            if detections:
                print(f"\n{Fore.GREEN}📍 帧 {current_frame_number}: 发现 {len(detections)} 个人物")
                
                for person in detections:
                    match_analysis = self.calculate_match_score(person, description_features)
                    
                    person_result = {
                        'frame_number': current_frame_number,
                        'person_id': person['id'],
                        'bbox': person['bbox'],
                        'confidence': person['confidence'],
                        'size': person['size'],
                        'match_analysis': match_analysis,
                        'frame': frame.copy()
                    }
                    
                    all_matches.append(person_result)
                    
                    print(f"  👤 人物 {person['id']}: 匹配分数 {match_analysis['total_score']:.3f}")
                    if match_analysis['matched_features']['colors']:
                        for color, info in match_analysis['matched_features']['colors'].items():
                            print(f"     {color}: {info['ratio']:.1%} ({info['region']})")
            
            frame_count += 1
        
        cap.release()
        
        # 排序并获取最佳匹配
        all_matches.sort(key=lambda x: x['match_analysis']['total_score'], reverse=True)
        
        # 保存结果
        result_summary = self.save_search_results(all_matches, description_features, output_dir)
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}🎯 搜索完成！")
        print(f"{Fore.GREEN}总检测: {len(all_matches)} 个人物")
        if all_matches:
            print(f"{Fore.GREEN}最佳匹配分数: {all_matches[0]['match_analysis']['total_score']:.3f}")
        print(f"{Fore.GREEN}结果保存在: {output_dir}")
        print(f"{Fore.CYAN}{'='*60}")
        
        return result_summary
    
    def save_search_results(self, all_matches: List[Dict], description_features: Dict, output_dir: str) -> Dict[str, Any]:
        """保存搜索结果"""
        if not all_matches:
            return {}
        
        # 保存前10个最佳匹配的可视化
        best_matches = all_matches[:10]
        
        print(f"\n{Fore.YELLOW}💾 保存最佳匹配结果...")
        
        for i, match in enumerate(best_matches):
            # 创建高质量可视化
            vis_frame = self.create_detailed_visualization(
                match['frame'], 
                [match], 
                description_features,
                f"匹配 #{i+1} - 分数: {match['match_analysis']['total_score']:.3f}"
            )
            
            # 保存图像
            filename = f"match_{i+1:02d}_frame_{match['frame_number']}_score_{match['match_analysis']['total_score']:.3f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, vis_frame)
            
            print(f"  ✅ 保存 {filename}")
        
        # 创建综合报告
        report = {
            'search_info': {
                'timestamp': datetime.now().isoformat(),
                'description': description_features['raw_text'],
                'parsed_features': description_features,
                'total_matches': len(all_matches)
            },
            'best_matches': []
        }
        
        for i, match in enumerate(best_matches):
            match_info = {
                'rank': i + 1,
                'frame_number': match['frame_number'],
                'person_id': match['person_id'],
                'bbox': match['bbox'],
                'size': match['size'],
                'confidence': match['confidence'],
                'match_score': match['match_analysis']['total_score'],
                'matched_features': match['match_analysis']['matched_features']
            }
            report['best_matches'].append(match_info)
        
        # 保存JSON报告
        with open(os.path.join(output_dir, 'search_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 创建汇总可视化
        self.create_summary_visualization(best_matches, description_features, output_dir)
        
        # 打印最佳结果
        print(f"\n{Fore.CYAN}🏆 最佳匹配结果:")
        for i, match in enumerate(best_matches[:5]):
            score = match['match_analysis']['total_score']
            print(f"  {i+1}. 帧{match['frame_number']} - 分数: {score:.3f}")
            if match['match_analysis']['matched_features']['colors']:
                for color, info in match['match_analysis']['matched_features']['colors'].items():
                    print(f"     {color}: {info['ratio']:.1%} 在 {info['region']}")
        
        return report
    
    def create_detailed_visualization(self, frame: np.ndarray, matches: List[Dict], 
                                   description_features: Dict, title: str) -> np.ndarray:
        """创建详细的可视化图像"""
        vis_frame = frame.copy()
        
        for match in matches:
            x1, y1, x2, y2 = match['bbox']
            score = match['match_analysis']['total_score']
            
            # 根据分数选择颜色
            if score > 0.7:
                color = (0, 255, 0)  # 绿色 - 高匹配
                thickness = 4
            elif score > 0.4:
                color = (0, 255, 255)  # 黄色 - 中等匹配
                thickness = 3
            else:
                color = (0, 100, 255)  # 橙色 - 低匹配
                thickness = 2
            
            # 绘制边框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签信息
            labels = [
                f"Person {match['person_id']}",
                f"Score: {score:.3f}",
                f"Conf: {match['confidence']:.3f}",
                f"Size: {match['size'][0]}x{match['size'][1]}"
            ]
            
            # 添加匹配的颜色信息
            if match['match_analysis']['matched_features']['colors']:
                for color_name, info in match['match_analysis']['matched_features']['colors'].items():
                    labels.append(f"{color_name}: {info['ratio']:.1%}")
            
            # 绘制信息背景
            label_height = len(labels) * 25 + 10
            label_width = 250
            cv2.rectangle(vis_frame, (x1, y1 - label_height), 
                         (x1 + label_width, y1), color, -1)
            
            # 绘制文字
            for i, label in enumerate(labels):
                y_pos = y1 - label_height + 20 + i * 25
                cv2.putText(vis_frame, label, (x1 + 5, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 可选信息条（默认关闭）
        SHOW_HEADER = False
        if SHOW_HEADER:
            info_bg_height = 120
            cv2.rectangle(vis_frame, (10, 10), (vis_frame.shape[1] - 10, info_bg_height), 
                         (50, 50, 50), -1)
            info_lines = [
                title,
                f"Target: {description_features['raw_text']}",
                f"Colors: {', '.join(description_features['colors'])}",
                f"Clothing: {', '.join(description_features['clothing'])}"
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(vis_frame, line, (20, 40 + i * 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def create_summary_visualization(self, best_matches: List[Dict], 
                                   description_features: Dict, output_dir: str):
        """创建汇总可视化"""
        if not best_matches:
            return
        
        # 创建matplotlib图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Summary - "{description_features["raw_text"]}"', 
                    fontsize=16, fontweight='bold')
        
        # 显示前6个最佳匹配
        for i in range(min(6, len(best_matches))):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            match = best_matches[i]
            frame = match['frame']
            x1, y1, x2, y2 = match['bbox']
            
            # 显示图像
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 添加边框
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # 设置标题（英文，避免中文字体警告）
            title = f"#{i+1}: Score {match['match_analysis']['total_score']:.3f}\n"
            title += f"Frame {match['frame_number']}, Person {match['person_id']}"
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(len(best_matches), 6):
            row = i // 3
            col = i % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存汇总可视化: summary_visualization.png")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 3:
        print(f"{Fore.CYAN}用法: python universal_person_finder.py <视频文件> <描述文本> [最大帧数]")
        print(f"{Fore.CYAN}示例:")
        print(f"  python universal_person_finder.py video.mp4 '红色头盔的人' 15")
        print(f"  python universal_person_finder.py video.mp4 'person with blue jacket' 20")
        print(f"  python universal_person_finder.py video.mp4 '穿工装的工人' 10")
        return
    
    video_path = sys.argv[1]
    description = sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    try:
        finder = UniversalPersonFinder()
        results = finder.find_best_matches(video_path, description, max_frames)
        
    except Exception as e:
        print(f"{Fore.RED}❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 