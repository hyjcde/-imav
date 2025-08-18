#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的多人俯视场景检测器
专门处理无人机拍摄的多人场景
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import logging
import os
from colorama import Fore, init

init(autoreset=True)
logger = logging.getLogger(__name__)

class SimpleMultiPersonDetector:
    """简化的多人俯视检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.detector = self._init_detector()
        
        # 俯视角度优化参数
        self.min_person_size = 15
        self.confidence_threshold = 0.25
        self.nms_threshold = 0.3
        
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
    
    def detect_multi_person_scene(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测多人场景"""
        detections = []
        original_height, original_width = frame.shape[:2]
        
        # 多尺度检测
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
                            
                            if self._is_valid_aerial_person(width, height, confidence):
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
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': all_scores[i],
                        'person_crop': all_crops[i],
                        'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                        'size': (int(x2 - x1), int(y2 - y1)),
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    detections.append(detection_info)
        
        # 简单的群体分析
        scene_analysis = self._simple_scene_analysis(detections, frame.shape)
        
        return {
            'detections': detections,
            'scene_analysis': scene_analysis
        }
    
    def _is_valid_aerial_person(self, width: float, height: float, confidence: float) -> bool:
        """判断是否为有效的俯视人物检测"""
        if width < self.min_person_size or height < self.min_person_size:
            return False
        
        aspect_ratio = width / height
        if not (0.3 <= aspect_ratio <= 0.8):
            return False
        
        area = width * height
        if area < 200 or area > 5000:
            return False
        
        return True
    
    def _simple_scene_analysis(self, detections: List[Dict], frame_shape: Tuple) -> Dict[str, Any]:
        """简单的场景分析"""
        if not detections:
            return {
                'person_count': 0,
                'density': 0,
                'clusters': [],
                'spatial_distribution': {}
            }
        
        height, width = frame_shape[:2]
        
        # 简单的空间分布分析
        centers = [det['center'] for det in detections]
        spatial_distribution = self._analyze_spatial_distribution(centers, width, height)
        
        # 简单的聚类（基于距离）
        clusters = self._simple_clustering(detections)
        
        frame_area = width * height
        density = len(detections) / (frame_area / 10000)
        
        return {
            'person_count': len(detections),
            'density': density,
            'clusters': clusters,
            'spatial_distribution': spatial_distribution
        }
    
    def _analyze_spatial_distribution(self, centers: List[Tuple], width: int, height: int) -> Dict[str, int]:
        """分析空间分布"""
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
        
        for x, y in centers:
            grid_x = min(int(x // grid_width), 2)
            grid_y = min(int(y // grid_height), 2)
            distribution[region_names[grid_y][grid_x]] += 1
        
        return distribution
    
    def _simple_clustering(self, detections: List[Dict]) -> List[Dict]:
        """简单的聚类分析"""
        clusters = []
        processed = set()
        
        for i, det1 in enumerate(detections):
            if i in processed:
                continue
            
            cluster = [det1]
            processed.add(i)
            
            for j, det2 in enumerate(detections):
                if j in processed or i == j:
                    continue
                
                # 计算距离
                center1 = det1['center']
                center2 = det2['center']
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < 80:  # 聚类距离阈值
                    cluster.append(det2)
                    processed.add(j)
            
            if len(cluster) >= 2:  # 至少2个人才算群体
                clusters.append({
                    'size': len(cluster),
                    'detections': cluster,
                    'center': np.mean([det['center'] for det in cluster], axis=0).tolist()
                })
        
        return clusters
    
    def extract_color_features(self, person_crop: np.ndarray) -> Dict[str, float]:
        """简单的颜色特征提取"""
        if person_crop is None or person_crop.size == 0:
            return {}
        
        # 分析头部区域（上1/3）
        height = person_crop.shape[0]
        head_region = person_crop[:height//3, :]
        
        if head_region.size == 0:
            return {}
        
        # 转换为HSV
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        total_pixels = head_region.shape[0] * head_region.shape[1]
        
        # HSV颜色范围
        color_ranges = {
            'red': [[(0, 120, 100), (8, 255, 255)], [(172, 120, 100), (180, 255, 255)]],
            'blue': [[(100, 120, 100), (130, 255, 255)]],
            'yellow': [[(20, 120, 100), (30, 255, 255)]],
            'green': [[(40, 120, 100), (80, 255, 255)]],
            'orange': [[(8, 120, 100), (20, 255, 255)]],
            'white': [[(0, 0, 200), (180, 55, 255)]],
            'black': [[(0, 0, 0), (180, 255, 50)]]
        }
        
        color_features = {}
        
        for color, ranges in color_ranges.items():
            total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                total_mask = cv2.bitwise_or(total_mask, mask)
            
            color_pixels = cv2.countNonZero(total_mask)
            ratio = color_pixels / total_pixels
            
            if ratio > 0.1:  # 至少占10%
                color_features[color] = ratio
        
        return color_features
    
    def find_target_person(self, frame: np.ndarray, description: str) -> Dict[str, Any]:
        """在多人场景中查找目标人物"""
        scene_result = self.detect_multi_person_scene(frame)
        detections = scene_result['detections']
        scene_analysis = scene_result['scene_analysis']
        
        if not detections:
            return {
                'status': 'no_persons_detected',
                'scene_analysis': scene_analysis,
                'matches': []
            }
        
        # 解析描述
        target_colors = self._parse_colors(description)
        
        # 查找匹配
        matches = []
        for detection in detections:
            color_features = self.extract_color_features(detection['person_crop'])
            match_score = self._calculate_match_score(color_features, target_colors, detection)
            
            if match_score > 0.2:
                matches.append((detection, match_score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'status': 'success' if matches else 'no_matches',
            'scene_analysis': scene_analysis,
            'matches': matches,
            'total_detections': len(detections)
        }
    
    def _parse_colors(self, description: str) -> List[str]:
        """解析颜色描述"""
        description = description.lower()
        colors = []
        
        color_keywords = {
            'red': ['红色', '红', 'red'],
            'blue': ['蓝色', '蓝', 'blue'],
            'yellow': ['黄色', '黄', 'yellow'],
            'green': ['绿色', '绿', 'green'],
            'white': ['白色', '白', 'white'],
            'black': ['黑色', '黑', 'black'],
            'orange': ['橙色', '橙', 'orange']
        }
        
        for color, keywords in color_keywords.items():
            if any(keyword in description for keyword in keywords):
                colors.append(color)
        
        return colors
    
    def _calculate_match_score(self, color_features: Dict[str, float], 
                             target_colors: List[str], detection: Dict) -> float:
        """计算匹配分数"""
        score = 0.0
        
        # 颜色匹配
        if target_colors:
            color_score = 0.0
            for target_color in target_colors:
                if target_color in color_features:
                    color_score += color_features[target_color]
            
            if target_colors:
                color_score /= len(target_colors)
            
            score += color_score * 0.7
        
        # 检测质量
        score += detection['confidence'] * 0.3
        
        return score
    
    def visualize_results(self, frame: np.ndarray, result: Dict, output_path: str = None):
        """可视化结果"""
        vis_frame = frame.copy()
        detections = result['scene_analysis']['person_count']
        matches = result.get('matches', [])
        
        # 绘制所有检测
        all_detections = []
        if 'scene_analysis' in result:
            # 从clusters中提取所有检测
            for cluster in result['scene_analysis']['clusters']:
                all_detections.extend(cluster['detections'])
            
            # 添加单独的人物
            for match in matches:
                detection = match[0]
                if detection not in all_detections:
                    all_detections.append(detection)
        
        # 绘制检测框
        for i, detection in enumerate(all_detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # 普通检测用蓝色
            color = (255, 0, 0)
            thickness = 2
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"P{i+1}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制匹配结果
        for i, (detection, score) in enumerate(matches[:3]):
            x1, y1, x2, y2 = detection['bbox']
            
            # 匹配用绿色
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            match_label = f"Match {i+1}: {score:.3f}"
            cv2.putText(vis_frame, match_label, (x1, y1-30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 添加统计信息
        scene_analysis = result['scene_analysis']
        stats = [
            f"Total: {scene_analysis['person_count']}",
            f"Groups: {len(scene_analysis['clusters'])}",
            f"Matches: {len(matches)}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(vis_frame, stat, (10, 30 + i*25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_frame)
        
        return vis_frame

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 3:
        video_path = "data/DJI_20250807124730_0002_S.MP4"
        description = "a man with red helmet"
        max_frames = 15
    else:
        video_path = sys.argv[1]
        description = sys.argv[2]
        max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    try:
        detector = SimpleMultiPersonDetector()
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}简化多人俯视场景分析")
        print(f"{Fore.CYAN}视频: {video_path}")
        print(f"{Fore.CYAN}目标: {description}")
        print(f"{Fore.CYAN}{'='*60}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{Fore.RED}无法打开视频文件")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, total_frames // max_frames)
        
        os.makedirs("outputs/simple_multi_results", exist_ok=True)
        
        frame_count = 0
        best_matches = []
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % skip_frames != 0:
                continue
            
            # 分析当前帧
            result = detector.find_target_person(frame, description)
            
            if result['scene_analysis']['person_count'] > 0:
                print(f"{Fore.GREEN}帧 {current_frame_number}:")
                print(f"  检测到 {result['scene_analysis']['person_count']} 个人物")
                print(f"  群体数: {len(result['scene_analysis']['clusters'])}")
                
                if result['matches']:
                    print(f"  找到 {len(result['matches'])} 个匹配项")
                    for i, (detection, score) in enumerate(result['matches'][:3]):
                        print(f"    匹配 {i+1}: 分数 {score:.3f}")
                    
                    # 保存最佳匹配
                    best_match = result['matches'][0]
                    best_matches.append((current_frame_number, best_match[1], result))
                
                # 可视化
                vis_frame = detector.visualize_results(
                    frame, result,
                    f"outputs/simple_multi_results/frame_{current_frame_number}.jpg"
                )
            
            frame_count += 1
        
        cap.release()
        
        # 显示最佳结果
        if best_matches:
            best_matches.sort(key=lambda x: x[1], reverse=True)
            print(f"\n{Fore.GREEN}最佳匹配结果:")
            for i, (frame_num, score, result) in enumerate(best_matches[:5]):
                print(f"  {i+1}. 帧 {frame_num}: 分数 {score:.3f}")
        
        print(f"\n{Fore.GREEN}分析完成！结果保存在: outputs/simple_multi_results/")
        
    except Exception as e:
        print(f"{Fore.RED}错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 