#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级场景分析器
集成YOLO检测和视觉语言模型，支持复杂描述理解
能够处理物体关系、姿态识别和空间位置理解
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
import torch
from PIL import Image
import re

# 尝试导入CLIP模型
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Installing...")

init(autoreset=True)
logger = logging.getLogger(__name__)

class AdvancedSceneAnalyzer:
    """高级场景分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_detector = self._init_yolo()
        self.clip_model = self._init_clip()
        
        # 检测参数（更低阈值以检出更多行人）
        self.min_person_size = 12
        self.confidence_threshold = 0.20
        self.nms_threshold = 0.4
        
        # COCO类别映射
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        
    def _init_yolo(self):
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
            logger.error(f"加载YOLO模型失败: {e}")
            raise
    
    def _init_clip(self):
        """初始化CLIP模型"""
        if not CLIP_AVAILABLE:
            print(f"{Fore.YELLOW}⚠️  CLIP模型不可用，将使用基础颜色分析")
            return None
        
        try:
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            print(f"{Fore.GREEN}✅ 成功加载CLIP模型")
            return {"model": model, "preprocess": preprocess}
        except Exception as e:
            print(f"{Fore.RED}❌ CLIP模型加载失败: {e}")
            return None
    
    def detect_all_objects(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """检测所有物体（不仅仅是人）"""
        detections = {"persons": [], "objects": []}
        original_height, original_width = frame.shape[:2]
        
        # 多尺度检测（人物召回为主，减少大尺度以降低车体干扰）
        scales = [0.85, 1.0, 1.2, 1.4]
        all_boxes = []
        all_scores = []
        all_classes = []
        all_crops = []
        
        for scale in scales:
            if scale != 1.0:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                scaled_frame = cv2.resize(frame, (new_width, new_height))
            else:
                scaled_frame = frame.copy()
            
            results = self.yolo_detector(scaled_frame,
                                       conf=self.confidence_threshold,
                                       iou=self.nms_threshold,
                                       verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
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
                        
                        # 加强人物几何过滤，降低车辆框进入后续流程
                        aspect = width / max(height, 1e-6)
                        area = width * height
                        is_reasonable_human = (0.2 <= aspect <= 1.5) and (150 <= area <= 20000)
                        if is_reasonable_human or class_id != 0:
                            all_boxes.append([x1, y1, x2, y2])
                            all_scores.append(confidence)
                            all_classes.append(class_id)
                            crop = frame[int(y1):int(y2), int(x1):int(x2)]
                            all_crops.append(crop)
        
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
                    class_id = all_classes[i]
                    confidence = all_scores[i]
                    crop = all_crops[i]
                    
                    # 仅保留与描述相关的物体类别（例如 umbrella/bench/suitcase/computer）
                    allowed_object_names = { 'umbrella', 'bench', 'suitcase', 'laptop', 'computer' }
                    class_name = self.coco_classes.get(class_id, f'class_{class_id}')
                    if class_id != 0 and not any(k in class_name for k in allowed_object_names):
                        continue
                    
                    detection_info = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'crop': crop,
                        'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                        'size': (int(x2 - x1), int(y2 - y1)),
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    
                    if class_id == 0:  # person
                        if self._is_valid_person(x2 - x1, y2 - y1, confidence):
                            detection_info['person_id'] = len(detections["persons"]) + 1
                            detections["persons"].append(detection_info)
                    else:
                        detection_info['object_id'] = len(detections["objects"]) + 1
                        detections["objects"].append(detection_info)
        
        return detections
    
    def _is_valid_person(self, width: float, height: float, confidence: float) -> bool:
        """判断是否为有效人物检测"""
        if width < self.min_person_size or height < self.min_person_size:
            return False
        
        aspect_ratio = width / height
        if not (0.2 <= aspect_ratio <= 1.5):
            return False
        
        area = width * height
        if area < 300 or area > 15000:
            return False
        
        return True
    
    def analyze_person_with_clip(self, person_crop: np.ndarray, description_parts: List[str]) -> Dict[str, float]:
        """使用CLIP分析人物特征"""
        if self.clip_model is None or person_crop is None or person_crop.size == 0:
            return {}
        
        try:
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            # 预处理图像
            image_input = self.clip_model["preprocess"](pil_image).unsqueeze(0).to(self.device)
            
            # 准备文本描述
            text_inputs = clip.tokenize(description_parts).to(self.device)
            
            # 计算特征
            with torch.no_grad():
                image_features = self.clip_model["model"].encode_image(image_input)
                text_features = self.clip_model["model"].encode_text(text_inputs)
                
                # 计算相似度
                similarities = torch.cosine_similarity(image_features, text_features, dim=1)
                similarities = similarities.cpu().numpy()
            
            # 返回每个描述部分的匹配分数
            result = {}
            for i, desc in enumerate(description_parts):
                if i < len(similarities):
                    result[desc] = float(similarities[i])
            
            return result
            
        except Exception as e:
            logger.error(f"CLIP分析失败: {e}")
            return {}
    
    def extract_color_features(self, person_crop: np.ndarray) -> Dict[str, Dict[str, float]]:
        """提取颜色特征（作为CLIP的补充）"""
        if person_crop is None or person_crop.size == 0:
            return {}
        
        height, width = person_crop.shape[:2]
        
        # 分区域分析
        regions = {
            'head': person_crop[:height//3, :],
            'torso': person_crop[height//3:2*height//3, :],
            'legs': person_crop[2*height//3:, :],
            'full': person_crop
        }
        
        # 扩展的HSV颜色范围
        color_ranges = {
            'red': [[(0, 120, 70), (10, 255, 255)], [(170, 120, 70), (180, 255, 255)]],
            'orange': [[(10, 120, 70), (25, 255, 255)]],
            'yellow': [[(20, 120, 70), (35, 255, 255)]],
            'green': [[(35, 120, 70), (85, 255, 255)]],
            'blue': [[(85, 120, 70), (125, 255, 255)]],
            'purple': [[(125, 120, 70), (155, 255, 255)]],
            'white': [[(0, 0, 200), (180, 30, 255)]],
            'black': [[(0, 0, 0), (180, 255, 70)]],
            'gray': [[(0, 0, 70), (180, 30, 200)]],
            'brown': [[(8, 50, 20), (25, 255, 200)]]
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
    
    def parse_complex_description(self, description: str) -> Dict[str, Any]:
        """解析复杂描述"""
        description_lower = description.lower()
        
        # 解析结果结构
        parsed = {
            'original': description,
            'colors': [],
            'clothing': [],
            'objects': [],
            'actions': [],
            'locations': [],
            'relations': [],
            'clip_descriptions': []  # 用于CLIP分析的描述片段
        }
        
        # 颜色关键词
        color_patterns = {
            'red': r'\b(red|crimson|scarlet)\b',
            'orange': r'\b(orange)\b',
            'yellow': r'\b(yellow|gold)\b',
            'green': r'\b(green|lime)\b',
            'blue': r'\b(blue|navy|cyan)\b',
            'purple': r'\b(purple|violet)\b',
            'white': r'\b(white|ivory)\b',
            'black': r'\b(black|dark)\b',
            'gray': r'\b(gray|grey|silver)\b',
            'brown': r'\b(brown|tan|beige)\b'
        }
        
        # 服装关键词
        clothing_patterns = {
            'jacket': r'\b(jacket|coat|blazer)\b',
            'helmet': r'\b(helmet|hard hat|construction helmet)\b',
            'shirt': r'\b(shirt|t-shirt|top|blouse)\b',
            'trousers': r'\b(trousers|pants|jeans)\b',
            'glasses': r'\b(glasses|sunglasses|spectacles)\b',
            'hat': r'\b(hat|cap)\b'
        }
        
        # 物体关键词
        object_patterns = {
            'umbrella': r'\b(umbrella)\b',
            'suitcase': r'\b(suitcase|luggage|bag)\b',
            'computer': r'\b(computer|laptop|notebook)\b',
            'bench': r'\b(bench|seat)\b'
        }
        
        # 动作关键词
        action_patterns = {
            'sitting': r'\b(sitting|seated)\b',
            'standing': r'\b(standing|upright)\b',
            'holding': r'\b(holding|carrying|with)\b',
            'wearing': r'\b(wearing|with)\b'
        }
        
        # 位置关键词
        location_patterns = {
            'on_grass': r'\bon.*grass\b',
            'on_bench': r'\bon.*bench\b',
            'next_to': r'\bnext to\b'
        }
        
        # 提取各种特征
        for color, pattern in color_patterns.items():
            if re.search(pattern, description_lower):
                parsed['colors'].append(color)
        
        for clothing, pattern in clothing_patterns.items():
            if re.search(pattern, description_lower):
                parsed['clothing'].append(clothing)
        
        for obj, pattern in object_patterns.items():
            if re.search(pattern, description_lower):
                parsed['objects'].append(obj)
        
        for action, pattern in action_patterns.items():
            if re.search(pattern, description_lower):
                parsed['actions'].append(action)
        
        for location, pattern in location_patterns.items():
            if re.search(pattern, description_lower):
                parsed['locations'].append(location)
        
        # 为CLIP准备描述片段
        parsed['clip_descriptions'] = [
            description,  # 完整描述
            f"person {' '.join(parsed['actions'])}",  # 动作描述
            f"person wearing {' '.join(parsed['colors'] + parsed['clothing'])}",  # 服装描述
            f"person with {' '.join(parsed['objects'])}"  # 物体描述
        ]
        
        # 移除空的描述
        parsed['clip_descriptions'] = [desc for desc in parsed['clip_descriptions'] if len(desc.split()) > 1]
        
        return parsed
    
    def find_scene_context(self, detections: Dict[str, List[Dict]], parsed_desc: Dict[str, Any]) -> Dict[str, Any]:
        """分析场景上下文"""
        scene_context = {
            'relevant_objects': [],
            'spatial_relations': [],
            'scene_score': 0.0
        }
        
        # 查找相关物体（仅限描述所需，过滤非相关类别如车辆）
        target_objects = parsed_desc['objects']
        ALLOWED = set(target_objects)
        for detection in detections['objects']:
            name = detection['class_name']
            if any(obj in name for obj in ALLOWED):
                scene_context['relevant_objects'].append(detection)
        
        # 计算场景匹配分数
        if target_objects:
            found_objects = len(scene_context['relevant_objects'])
            scene_context['scene_score'] = found_objects / len(target_objects)
        
        return scene_context
    
    def calculate_advanced_match_score(self, person: Dict, scene_context: Dict, 
                                     parsed_desc: Dict, clip_scores: Dict) -> Dict[str, Any]:
        """计算高级匹配分数"""
        # 提取传统颜色特征
        color_features = self.extract_color_features(person['crop'])
        
        match_analysis = {
            'clip_scores': clip_scores,
            'color_features': color_features,
            'scene_context': scene_context,
            'component_scores': {
                'clip_score': 0.0,
                'color_score': 0.0,
                'scene_score': scene_context['scene_score'],
                'quality_score': person['confidence']
            },
            'total_score': 0.0,
            'explanation': []
        }
        
        # CLIP分数 (主要评分)
        if clip_scores:
            clip_score = np.mean(list(clip_scores.values()))
            match_analysis['component_scores']['clip_score'] = clip_score
            match_analysis['explanation'].append(f"CLIP similarity: {clip_score:.3f}")
        
        # 颜色匹配分数 (辅助评分)
        color_matches = 0
        total_color_score = 0.0
        target_colors = parsed_desc['colors']
        
        if target_colors:
            for target_color in target_colors:
                best_match = 0.0
                best_region = None
                
                for region, colors in color_features.items():
                    if target_color in colors:
                        if colors[target_color] > best_match:
                            best_match = colors[target_color]
                            best_region = region
                
                if best_match > 0.1:
                    color_matches += 1
                    total_color_score += min(best_match * 2, 1.0)
                    match_analysis['explanation'].append(
                        f"{target_color}: {best_match:.1%} in {best_region}"
                    )
            
            if color_matches > 0:
                match_analysis['component_scores']['color_score'] = total_color_score / len(target_colors)
        
        # 综合评分
        weights = {
            'clip': 0.6 if clip_scores else 0.0,
            'color': 0.3 if target_colors else 0.0,
            'scene': 0.1 if scene_context['scene_score'] > 0 else 0.0,
            'quality': 0.1
        }
        
        # 重新分配权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        
        match_analysis['total_score'] = (
            match_analysis['component_scores']['clip_score'] * weights['clip'] +
            match_analysis['component_scores']['color_score'] * weights['color'] +
            match_analysis['component_scores']['scene_score'] * weights['scene'] +
            match_analysis['component_scores']['quality_score'] * weights['quality']
        )
        
        return match_analysis
    
    def analyze_complex_scene(self, video_path: str, description: str, max_frames: int = 15) -> Dict[str, Any]:
        """分析复杂场景"""
        print(f"\n{Fore.CYAN}{'='*90}")
        print(f"{Fore.CYAN}🧠 高级场景分析器")
        print(f"{Fore.CYAN}视频: {video_path}")
        print(f"{Fore.CYAN}复杂描述: {Style.BRIGHT}{description}")
        print(f"{Fore.CYAN}{'='*90}\n")
        
        # 解析复杂描述
        parsed_desc = self.parse_complex_description(description)
        print(f"{Fore.YELLOW}📝 智能解析结果:")
        print(f"   颜色: {parsed_desc['colors']}")
        print(f"   服装: {parsed_desc['clothing']}")
        print(f"   物体: {parsed_desc['objects']}")
        print(f"   动作: {parsed_desc['actions']}")
        print(f"   位置: {parsed_desc['locations']}")
        print(f"   CLIP描述片段: {len(parsed_desc['clip_descriptions'])}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{Fore.RED}❌ 无法打开视频文件")
            return {}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, total_frames // max_frames)
        
        output_dir = os.path.join("outputs", f"advanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        
        all_matches = []
        detection_stats = []
        frame_count = 0
        
        print(f"\n{Fore.YELLOW}🔍 开始高级分析 {max_frames} 帧...")
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % skip_frames != 0:
                continue
            
            # 检测所有物体
            detections = self.detect_all_objects(frame)

            # 总是先输出“所有行人”的一张图
            detection_stats.append({'frame': current_frame_number, 'num_persons': len(detections['persons'])})
            if detections['persons']:
                vis_all = frame.copy()
                for p in detections['persons']:
                    x1, y1, x2, y2 = p['bbox']
                    cv2.rectangle(vis_all, (x1, y1), (x2, y2), (220, 220, 220), 2)
                    pid = p.get('person_id', 0)
                    cv2.putText(vis_all, f"P{pid}", (x1+3, max(y1-6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.imwrite(os.path.join(output_dir, f"all_persons_frame_{current_frame_number}.jpg"), vis_all)

            if detections["persons"]:
                print(f"\n{Fore.GREEN}📍 帧 {current_frame_number}: 发现 {len(detections['persons'])} 个人物, {len(detections['objects'])} 个物体")

                # 分析场景上下文
                scene_context = self.find_scene_context(detections, parsed_desc)

                for person in detections["persons"]:
                    # CLIP分析
                    clip_scores = {}
                    if self.clip_model and parsed_desc['clip_descriptions']:
                        clip_scores = self.analyze_person_with_clip(
                            person['crop'], 
                            parsed_desc['clip_descriptions']
                        )

                    # 计算高级匹配分数
                    match_analysis = self.calculate_advanced_match_score(
                        person, scene_context, parsed_desc, clip_scores
                    )

                    person_result = {
                        'frame_number': current_frame_number,
                        'person_id': person['person_id'],
                        'bbox': person['bbox'],
                        'confidence': person['confidence'],
                        'size': person['size'],
                        'match_analysis': match_analysis,
                        'frame': frame.copy(),
                        'detections': detections  # 保存完整检测结果
                    }

                    all_matches.append(person_result)

                    print(f"  👤 人物 {person['person_id']}: 总分 {match_analysis['total_score']:.3f}")
                    if clip_scores:
                        print(f"     CLIP分析: {clip_scores}")
                    if match_analysis['explanation']:
                        for exp in match_analysis['explanation']:
                            print(f"     {exp}")
            
            frame_count += 1
        
        cap.release()
        
        # 排序并保存结果
        all_matches.sort(key=lambda x: x['match_analysis']['total_score'], reverse=True)
        
        result_summary = self.save_advanced_results(all_matches, parsed_desc, output_dir)
        # 保存检测统计
        with open(os.path.join(output_dir, 'all_persons_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(detection_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.GREEN}🎯 高级分析完成！")
        print(f"{Fore.GREEN}总检测: {len(all_matches)} 个人物")
        if all_matches:
            print(f"{Fore.GREEN}最佳匹配分数: {all_matches[0]['match_analysis']['total_score']:.3f}")
        print(f"{Fore.GREEN}结果保存在: {output_dir}")
        print(f"{Fore.CYAN}{'='*70}")
        
        return result_summary
    
    def save_advanced_results(self, all_matches: List[Dict], parsed_desc: Dict, output_dir: str) -> Dict[str, Any]:
        """保存高级分析结果"""
        if not all_matches:
            return {}
        
        best_matches = all_matches[:10]
        
        print(f"\n{Fore.YELLOW}💾 保存高级分析结果...")
        
        for i, match in enumerate(best_matches):
            # 创建详细可视化
            vis_frame = self.create_advanced_visualization(
                match['frame'],
                match,
                parsed_desc,
                f"高级匹配 #{i+1} - 分数: {match['match_analysis']['total_score']:.3f}"
            )
            
            filename = f"advanced_match_{i+1:02d}_frame_{match['frame_number']}_score_{match['match_analysis']['total_score']:.3f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, vis_frame)
            
            print(f"  ✅ 保存 {filename}")
        
        # 创建分析报告
        report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'description': parsed_desc['original'],
                'parsed_features': parsed_desc,
                'total_matches': len(all_matches),
                'clip_available': self.clip_model is not None
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
                'total_score': match['match_analysis']['total_score'],
                'component_scores': match['match_analysis']['component_scores'],
                'clip_scores': match['match_analysis']['clip_scores'],
                'explanation': match['match_analysis']['explanation']
            }
            report['best_matches'].append(match_info)
        
        with open(os.path.join(output_dir, 'advanced_analysis_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印最佳结果
        print(f"\n{Fore.CYAN}🏆 最佳匹配结果:")
        for i, match in enumerate(best_matches[:5]):
            score = match['match_analysis']['total_score']
            print(f"  {i+1}. 帧{match['frame_number']} - 总分: {score:.3f}")
            if match['match_analysis']['clip_scores']:
                print(f"     CLIP: {list(match['match_analysis']['clip_scores'].values())}")
            for exp in match['match_analysis']['explanation'][:2]:  # 只显示前2个解释
                print(f"     {exp}")
        
        return report
    
    def create_advanced_visualization(self, frame: np.ndarray, match: Dict, 
                                     parsed_desc: Dict, title: str) -> np.ndarray:
         """创建高级可视化"""
         vis_frame = frame.copy()
         
         # 先把本帧所有检测到的行人标出来（细框、浅色）
         if 'detections' in match and 'persons' in match['detections']:
             for p in match['detections']['persons']:
                 px1, py1, px2, py2 = p['bbox']
                 cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (220, 220, 220), 2)
                 pid = p.get('person_id', 0)
                 label = f"P{pid}"
                 tsize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                 cv2.rectangle(vis_frame, (px1, py1 - tsize[1] - 6), (px1 + tsize[0] + 6, py1), (220, 220, 220), -1)
                 cv2.putText(vis_frame, label, (px1 + 3, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
         
         # 绘制人物检测框
         person_bbox = match['bbox']
         x1, y1, x2, y2 = person_bbox
         score = match['match_analysis']['total_score']
         
         # 根据分数选择颜色
         if score > 0.7:
             color = (0, 255, 0)  # 绿色
             thickness = 4
         elif score > 0.4:
             color = (0, 255, 255)  # 黄色
             thickness = 3
         else:
             color = (0, 100, 255)  # 橙色
             thickness = 2
         
         cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
         
         # 绘制相关物体（仅与描述相关，避免车辆等干扰）
         DRAW_OBJECTS = 'relevant'  # 'none'|'relevant'
         if DRAW_OBJECTS != 'none' and 'match_analysis' in match:
             relevant = match['match_analysis'].get('scene_context', {}).get('relevant_objects', [])
             for obj in relevant:
                 ox1, oy1, ox2, oy2 = obj['bbox']
                 label = obj.get('class_name', 'obj')
                 cv2.rectangle(vis_frame, (ox1, oy1), (ox2, oy2), (255, 0, 150), 2)
                 cv2.putText(vis_frame, label, (ox1, max(oy1-5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 150), 1)
         
         # 可选信息条（默认关闭）
         SHOW_HEADER = False
         if SHOW_HEADER:
             info_lines = [
                 title,
                 f"Target: {parsed_desc['original'][:50]}...",
                 f"CLIP Available: {'Yes' if self.clip_model else 'No'}",
                 f"Components: C:{match['match_analysis']['component_scores']['clip_score']:.2f} "
                 f"Col:{match['match_analysis']['component_scores']['color_score']:.2f} "
                 f"Sc:{match['match_analysis']['component_scores']['scene_score']:.2f}"
             ]
             info_height = len(info_lines) * 30 + 20
             cv2.rectangle(vis_frame, (10, 10), (vis_frame.shape[1] - 10, info_height), (50, 50, 50), -1)
             for i, line in enumerate(info_lines):
                 cv2.putText(vis_frame, line, (20, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
         
         return vis_frame

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 3:
        print(f"{Fore.CYAN}用法: python advanced_scene_analyzer.py <视频文件> <复杂描述> [最大帧数]")
        print(f"{Fore.CYAN}示例:")
        print(f'  python advanced_scene_analyzer.py video.mp4 "Find the person with an orange jacket and a yellow construction helmet" 15')
        print(f'  python advanced_scene_analyzer.py video.mp4 "person sitting on a bench holding an umbrella" 12')
        return
    
    video_path = sys.argv[1]
    description = sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    try:
        analyzer = AdvancedSceneAnalyzer()
        results = analyzer.analyze_complex_scene(video_path, description, max_frames)
        
    except Exception as e:
        print(f"{Fore.RED}❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 