#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多人俯视场景检测系统
专门处理无人机拍摄的多人场景，考虑空间关系和群体行为
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import logging
from enhanced_color_analysis import EnhancedColorAnalyzer
import os
import matplotlib.pyplot as plt
from colorama import Fore, init
from sklearn.cluster import DBSCAN
import math

init(autoreset=True)
logger = logging.getLogger(__name__)

class MultiPersonAerialDetector:
    """多人俯视场景检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.detector = self._init_detector()
        self.color_analyzer = EnhancedColorAnalyzer()
        
        # 俯视角度优化参数
        self.min_person_size = 15  # 进一步降低最小尺寸
        self.confidence_threshold = 0.25  # 降低置信度以检测更多小目标
        self.nms_threshold = 0.3  # 降低NMS阈值，允许更近距离的检测
        
        # 群体分析参数
        self.clustering_eps = 50  # DBSCAN聚类距离阈值（像素）
        self.min_samples = 2  # 最小聚类样本数
        
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
                # 使用更大的模型
                model = YOLO('weights/yolov8m.pt')  # 升级到medium模型
                logger.info("成功加载YOLO模型: yolov8m.pt")
                return model
            finally:
                torch.load = original_load
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def detect_multi_person_scene(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        检测多人场景，返回详细的空间和群体信息
        """
        detections = []
        original_height, original_width = frame.shape[:2]
        
        # 多尺度检测 - 针对俯视角度优化
        scales = [0.8, 1.0, 1.3, 1.6]  # 更多尺度，包括缩小检测
        
        all_boxes = []
        all_scores = []
        all_crops = []
        
        for scale in scales:
            # 缩放图像
            if scale != 1.0:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                scaled_frame = cv2.resize(frame, (new_width, new_height))
            else:
                scaled_frame = frame.copy()
            
            # YOLO检测 - 更低的置信度
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
                            
                            # 调整回原始尺寸
                            if scale != 1.0:
                                x1, x2 = x1 / scale, x2 / scale
                                y1, y2 = y1 / scale, y2 / scale
                            
                            # 边界检查
                            x1 = max(0, min(x1, original_width))
                            y1 = max(0, min(y1, original_height))
                            x2 = max(0, min(x2, original_width))
                            y2 = max(0, min(y2, original_height))
                            
                            width = x2 - x1
                            height = y2 - y1
                            
                            # 俯视角度特征检查
                            if self._is_valid_aerial_person(width, height, confidence):
                                all_boxes.append([x1, y1, x2, y2])
                                all_scores.append(confidence)
                                
                                person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                all_crops.append(person_crop)
        
        # 改进的NMS处理
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
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': all_scores[i],
                        'person_crop': all_crops[i],
                        'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                        'size': (int(x2 - x1), int(y2 - y1)),
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    detections.append(detection_info)
        
        # 分析多人场景
        scene_analysis = self._analyze_multi_person_scene(detections, frame.shape)
        
        return {
            'detections': detections,
            'scene_analysis': scene_analysis,
            'frame_shape': frame.shape
        }
    
    def _is_valid_aerial_person(self, width: float, height: float, confidence: float) -> bool:
        """
        判断是否为有效的俯视人物检测
        参考PyImageSearch的无人机目标检测方法
        """
        # 基本尺寸检查
        if width < self.min_person_size or height < self.min_person_size:
            return False
        
        # 俯视角度的宽高比检查（人物在俯视角度下通常更"瘦"）
        aspect_ratio = width / height
        if not (0.3 <= aspect_ratio <= 0.8):  # 俯视人物的典型宽高比
            return False
        
        # 面积检查（太小或太大都不太可能是人物）
        area = width * height
        if area < 200 or area > 5000:  # 根据俯视角度调整
            return False
        
        return True
    
    def _analyze_multi_person_scene(self, detections: List[Dict], frame_shape: Tuple) -> Dict[str, Any]:
        """
        分析多人场景的空间关系和群体特征
        """
        if not detections:
            return {
                'person_count': 0,
                'density': 0,
                'clusters': [],
                'spatial_distribution': {}
            }
        
        # 提取人物中心点
        centers = np.array([det['center'] for det in detections])
        
        # 使用DBSCAN进行群体聚类
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(centers)
        
        # 分析聚类结果
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声点（孤立的人）
                continue
                
            cluster_mask = cluster_labels == label
            cluster_centers = centers[cluster_mask]
            cluster_detections = [det for i, det in enumerate(detections) if cluster_mask[i]]
            
            # 计算聚类统计信息
            cluster_info = {
                'label': label,
                'size': len(cluster_detections),
                'center': np.mean(cluster_centers, axis=0).tolist(),
                'bounding_box': self._get_cluster_bbox(cluster_centers),
                'avg_confidence': np.mean([det['confidence'] for det in cluster_detections]),
                'detections': cluster_detections
            }
            clusters.append(cluster_info)
        
        # 空间分布分析
        height, width = frame_shape[:2]
        spatial_distribution = self._analyze_spatial_distribution(centers, width, height)
        
        # 密度计算
        frame_area = width * height
        density = len(detections) / (frame_area / 10000)  # 每万像素的人数
        
        return {
            'person_count': len(detections),
            'density': density,
            'clusters': clusters,
            'isolated_persons': int(np.sum(cluster_labels == -1)),
            'spatial_distribution': spatial_distribution
        }
    
    def _get_cluster_bbox(self, centers: np.ndarray) -> List[int]:
        """计算聚类的边界框"""
        min_x, min_y = np.min(centers, axis=0)
        max_x, max_y = np.max(centers, axis=0)
        return [int(min_x), int(min_y), int(max_x), int(max_y)]
    
    def _analyze_spatial_distribution(self, centers: np.ndarray, width: int, height: int) -> Dict[str, int]:
        """分析人物的空间分布"""
        # 将图像分为9个区域（3x3网格）
        grid_width = width // 3
        grid_height = height // 3
        
        distribution = {
            'top_left': 0, 'top_center': 0, 'top_right': 0,
            'middle_left': 0, 'middle_center': 0, 'middle_right': 0,
            'bottom_left': 0, 'bottom_center': 0, 'bottom_right': 0
        }
        
        region_names = [
            ['top_left', 'top_center', 'top_right'],
            ['middle_left', 'middle_center', 'middle_right'],
            ['bottom_left', 'bottom_center', 'bottom_right']
        ]
        
        for center in centers:
            x, y = center
            grid_x = min(int(x // grid_width), 2)
            grid_y = min(int(y // grid_height), 2)
            distribution[region_names[grid_y][grid_x]] += 1
        
        return distribution
    
    def find_target_person_in_multi_scene(self, frame: np.ndarray, description: str) -> Dict[str, Any]:
        """
        在多人场景中查找目标人物
        """
        # 检测多人场景
        scene_result = self.detect_multi_person_scene(frame)
        detections = scene_result['detections']
        scene_analysis = scene_result['scene_analysis']
        
        if not detections:
            return {
                'status': 'no_persons_detected',
                'scene_analysis': scene_analysis,
                'matches': []
            }
        
        # 使用增强颜色分析器查找头盔候选
        # 为每个检测添加person_crop字段以兼容颜色分析器
        formatted_detections = []
        for detection in detections:
            formatted_detection = detection.copy()
            if 'person_crop' not in formatted_detection:
                formatted_detection['person_crop'] = detection.get('person_crop')
            formatted_detections.append(formatted_detection)
        
        helmet_candidates = self.color_analyzer.find_helmet_candidates(formatted_detections)
        
        # 解析文本描述
        target_features = self._parse_description(description)
        
        # 匹配候选
        matches = []
        for candidate in helmet_candidates:
            match_score = self._calculate_multi_scene_score(
                candidate, target_features, scene_analysis
            )
            
            if match_score > 0.2:  # 降低阈值以适应俯视角度
                matches.append((candidate, match_score))
        
        # 按分数排序
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'status': 'success' if matches else 'no_matches',
            'scene_analysis': scene_analysis,
            'matches': matches,
            'total_detections': len(detections)
        }
    
    def _parse_description(self, description: str) -> Dict[str, Any]:
        """解析文本描述"""
        description = description.lower()
        
        # 颜色关键词
        color_keywords = {
            'red': ['红色', '红', 'red'],
            'blue': ['蓝色', '蓝', 'blue'],
            'yellow': ['黄色', '黄', 'yellow'],
            'green': ['绿色', '绿', 'green'],
            'white': ['白色', '白', 'white'],
            'black': ['黑色', '黑', 'black'],
            'orange': ['橙色', '橙', 'orange']
        }
        
        # 服装关键词
        clothing_keywords = {
            'helmet': ['头盔', '安全帽', 'helmet'],
            'hat': ['帽子', '帽', 'hat'],
            'shirt': ['上衣', '衬衫', 'shirt']
        }
        
        # 位置关键词（针对多人场景）
        position_keywords = {
            'left': ['左', '左边', 'left'],
            'right': ['右', '右边', 'right'],
            'center': ['中间', '中央', 'center', 'middle'],
            'alone': ['单独', '孤立', 'alone', 'isolated'],
            'group': ['群体', '一群', 'group', 'crowd']
        }
        
        features = {'colors': [], 'clothing': [], 'position': []}
        
        # 提取特征
        for color, keywords in color_keywords.items():
            if any(keyword in description for keyword in keywords):
                features['colors'].append(color)
        
        for clothing, keywords in clothing_keywords.items():
            if any(keyword in description for keyword in keywords):
                features['clothing'].append(clothing)
        
        for position, keywords in position_keywords.items():
            if any(keyword in description for keyword in keywords):
                features['position'].append(position)
        
        return features
    
    def _calculate_multi_scene_score(self, candidate: Dict, target_features: Dict, 
                                   scene_analysis: Dict) -> float:
        """
        计算多人场景中的匹配分数
        """
        person = candidate['person']
        color_features = candidate['color_features']
        helmet_score = candidate['helmet_score']
        
        score = 0.0
        total_weight = 0.0
        
        # 头盔/颜色匹配（主要权重）
        if target_features['colors']:
            color_weight = 0.6
            total_weight += color_weight
            
            color_match = 0.0
            for target_color in target_features['colors']:
                # 检查头部区域颜色
                head_color_key = f"head_{target_color}"
                if head_color_key in color_features:
                    color_match += color_features[head_color_key]
                
                # 检查整体颜色
                if target_color in color_features:
                    color_match += color_features[target_color] * 0.5
            
            score += (color_match + helmet_score) * color_weight
        
        # 位置匹配（针对多人场景）
        if target_features['position']:
            position_weight = 0.3
            total_weight += position_weight
            
            position_score = self._calculate_position_score(
                person, target_features['position'], scene_analysis
            )
            score += position_score * position_weight
        
        # 基础检测质量
        quality_weight = 0.1
        total_weight += quality_weight
        quality_score = person['confidence']
        score += quality_score * quality_weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_position_score(self, person: Dict, position_features: List[str], 
                                scene_analysis: Dict) -> float:
        """计算位置匹配分数"""
        person_center = person['center']
        spatial_dist = scene_analysis['spatial_distribution']
        
        score = 0.0
        
        for pos_feature in position_features:
            if pos_feature == 'alone' or pos_feature == 'isolated':
                # 检查是否为孤立的人
                if scene_analysis['isolated_persons'] > 0:
                    score += 0.8
            elif pos_feature == 'group':
                # 检查是否在群体中
                if scene_analysis['clusters']:
                    score += 0.6
            elif pos_feature in ['left', 'right', 'center']:
                # 基于空间分布的位置匹配
                if pos_feature == 'left':
                    left_count = (spatial_dist.get('top_left', 0) + 
                                spatial_dist.get('middle_left', 0) + 
                                spatial_dist.get('bottom_left', 0))
                    if left_count > 0:
                        score += 0.5
                # 类似处理其他位置...
        
        return min(score, 1.0)
    
    def visualize_multi_person_scene(self, frame: np.ndarray, scene_result: Dict, 
                                   matches: List = None, output_path: str = None):
        """
        可视化多人场景分析结果
        """
        vis_frame = frame.copy()
        detections = scene_result['detections']
        scene_analysis = scene_result['scene_analysis']
        
        # 绘制所有检测框
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # 默认颜色（蓝色）
            color = (255, 0, 0)
            thickness = 1
            
            # 绘制检测框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # 添加标签
            label = f"Person {i+1}\n{confidence:.2f}"
            label_lines = label.split('\n')
            
            for j, line in enumerate(label_lines):
                y_pos = y1 - 10 - j * 15
                cv2.putText(vis_frame, line, (x1, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 绘制匹配结果
        if matches:
            for i, (candidate, score) in enumerate(matches[:3]):  # 显示前3个匹配
                person = candidate['person']
                x1, y1, x2, y2 = person['bbox']
                
                # 匹配框用绿色
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # 匹配标签
                match_label = f"Match {i+1}: {score:.3f}"
                cv2.putText(vis_frame, match_label, (x1, y1 - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制聚类信息
        for cluster in scene_analysis['clusters']:
            bbox = cluster['bounding_box']
            x1, y1, x2, y2 = bbox
            
            # 聚类边界框（黄色虚线）
            self._draw_dashed_rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # 聚类标签
            cluster_label = f"Group {cluster['label']}: {cluster['size']} people"
            cv2.putText(vis_frame, cluster_label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 添加场景统计信息
        stats_text = [
            f"Total Persons: {scene_analysis['person_count']}",
            f"Groups: {len(scene_analysis['clusters'])}",
            f"Isolated: {scene_analysis['isolated_persons']}",
            f"Density: {scene_analysis['density']:.3f}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_frame)
        
        return vis_frame
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """绘制虚线矩形"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # 绘制虚线边框
        dash_length = 10
        
        # 顶边和底边
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # 左边和右边
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 3:
        video_path = "data/DJI_20250807124730_0002_S.MP4"
        description = "a man with red helmet"
        max_frames = 20
        print(f"{Fore.CYAN}使用默认参数:")
        print(f"  视频: {video_path}")
        print(f"  描述: {description}")
    else:
        video_path = sys.argv[1]
        description = sys.argv[2]
        max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    try:
        detector = MultiPersonAerialDetector()
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}多人俯视场景分析")
        print(f"{Fore.CYAN}视频: {video_path}")
        print(f"{Fore.CYAN}目标: {description}")
        print(f"{Fore.CYAN}{'='*60}\n")
        
        # 处理视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{Fore.RED}无法打开视频文件")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, total_frames // max_frames)
        
        os.makedirs("outputs/multi_person_results", exist_ok=True)
        
        frame_count = 0
        results = []
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % skip_frames != 0:
                continue
            
            # 分析多人场景
            result = detector.find_target_person_in_multi_scene(frame, description)
            
            if result['scene_analysis']['person_count'] > 0:
                print(f"{Fore.GREEN}帧 {current_frame_number}:")
                print(f"  检测到 {result['scene_analysis']['person_count']} 个人物")
                print(f"  群体数: {len(result['scene_analysis']['clusters'])}")
                print(f"  孤立人数: {result['scene_analysis']['isolated_persons']}")
                
                if result['matches']:
                    print(f"  找到 {len(result['matches'])} 个匹配项")
                    for i, (candidate, score) in enumerate(result['matches'][:3]):
                        print(f"    匹配 {i+1}: 分数 {score:.3f}")
                
                # 可视化并保存
                vis_frame = detector.visualize_multi_person_scene(
                    frame, result, result['matches'],
                    f"outputs/multi_person_results/frame_{current_frame_number}_analysis.jpg"
                )
                
                results.append((current_frame_number, result))
            
            frame_count += 1
        
        cap.release()
        
        print(f"\n{Fore.GREEN}分析完成！共处理 {len(results)} 帧包含人物的场景")
        print(f"{Fore.GREEN}结果保存在: outputs/multi_person_results/")
        
    except Exception as e:
        print(f"{Fore.RED}错误: {e}")

if __name__ == "__main__":
    main() 