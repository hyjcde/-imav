#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本描述匹配模块
将文本描述转换为可匹配的视觉特征
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class TextMatcher:
    """文本描述匹配器类"""
    
    def __init__(self):
        """初始化文本匹配器"""
        # 定义颜色关键词映射
        self.color_keywords = {
            'red': ['红色', '红', 'red'],
            'blue': ['蓝色', '蓝', 'blue'],
            'green': ['绿色', '绿', 'green'],
            'yellow': ['黄色', '黄', 'yellow'],
            'black': ['黑色', '黑', 'black'],
            'white': ['白色', '白', 'white'],
            'orange': ['橙色', '橙', 'orange'],
            'purple': ['紫色', '紫', 'purple'],
            'gray': ['灰色', '灰', 'gray', 'grey'],
            'brown': ['棕色', '棕', '褐色', 'brown']
        }
        
        # 定义服装关键词
        self.clothing_keywords = {
            'helmet': ['头盔', '安全帽', 'helmet'],
            'hat': ['帽子', '帽', 'hat', 'cap'],
            'shirt': ['衬衫', '上衣', 'shirt'],
            'jacket': ['外套', '夹克', 'jacket'],
            'pants': ['裤子', '长裤', 'pants', 'trousers'],
            'shorts': ['短裤', 'shorts'],
            'dress': ['裙子', '连衣裙', 'dress'],
            'coat': ['大衣', '外衣', 'coat']
        }
        
        # 定义位置关键词
        self.position_keywords = {
            'left': ['左', '左边', 'left'],
            'right': ['右', '右边', 'right'],
            'center': ['中间', '中央', 'center', 'middle'],
            'front': ['前面', '前', 'front'],
            'back': ['后面', '后', 'back']
        }
        
        # HSV颜色范围定义
        self.hsv_color_ranges = {
            'red': [(0, 120, 70), (10, 255, 255)],  # 红色范围1
            'red2': [(170, 120, 70), (180, 255, 255)],  # 红色范围2
            'blue': [(100, 150, 0), (124, 255, 255)],
            'green': [(25, 52, 72), (102, 255, 255)],
            'yellow': [(15, 150, 150), (35, 255, 255)],
            'orange': [(5, 150, 150), (15, 255, 255)],
            'purple': [(125, 150, 0), (150, 255, 255)],
            'black': [(0, 0, 0), (180, 255, 30)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'gray': [(0, 0, 50), (180, 30, 200)]
        }
    
    def parse_description(self, description: str) -> Dict[str, Any]:
        """
        解析文本描述，提取关键特征
        
        Args:
            description (str): 文本描述
            
        Returns:
            Dict[str, Any]: 解析后的特征字典
        """
        description = description.lower()
        features = {
            'colors': [],
            'clothing': [],
            'position': [],
            'raw_text': description
        }
        
        # 提取颜色信息
        for color, keywords in self.color_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    features['colors'].append(color)
                    break
        
        # 提取服装信息
        for clothing, keywords in self.clothing_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    features['clothing'].append(clothing)
                    break
        
        # 提取位置信息
        for position, keywords in self.position_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    features['position'].append(position)
                    break
        
        logger.info(f"解析描述 '{description}' -> 特征: {features}")
        return features
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        从图像中提取颜色特征
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            Dict[str, float]: 颜色特征字典，包含各颜色的占比
        """
        if image is None or image.size == 0:
            return {}
            
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total_pixels = image.shape[0] * image.shape[1]
        
        color_ratios = {}
        
        for color, (lower, upper) in self.hsv_color_ranges.items():
            # 创建颜色掩码
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # 特殊处理红色（需要两个范围）
            if color == 'red':
                lower2, upper2 = self.hsv_color_ranges['red2']
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask, mask2)
            
            # 计算颜色占比
            color_pixels = cv2.countNonZero(mask)
            ratio = color_pixels / total_pixels
            
            # 只保留有意义的颜色占比
            if ratio > 0.05:  # 至少占5%
                color_ratios[color.replace('2', '')] = ratio
        
        return color_ratios
    
    def calculate_match_score(self, person_features: Dict[str, Any], 
                            target_features: Dict[str, Any]) -> float:
        """
        计算人物特征与目标描述的匹配分数
        
        Args:
            person_features (Dict): 人物的视觉特征
            target_features (Dict): 目标描述的特征
            
        Returns:
            float: 匹配分数 (0-1)
        """
        score = 0.0
        total_weight = 0.0
        
        # 颜色匹配（权重最高）
        if target_features['colors']:
            color_weight = 0.6
            total_weight += color_weight
            
            color_score = 0.0
            for target_color in target_features['colors']:
                if target_color in person_features.get('colors', {}):
                    # 颜色占比越高，分数越高
                    color_ratio = person_features['colors'][target_color]
                    color_score += min(color_ratio * 2, 1.0)  # 最高1分
            
            if target_features['colors']:
                color_score /= len(target_features['colors'])
            
            score += color_score * color_weight
            logger.debug(f"颜色匹配分数: {color_score:.3f}")
        
        # 服装匹配（中等权重）
        if target_features['clothing']:
            clothing_weight = 0.3
            total_weight += clothing_weight
            
            # 这里可以扩展为更复杂的服装检测
            # 目前简化为存在性检查
            clothing_score = 0.5  # 给予基础分数
            score += clothing_score * clothing_weight
            logger.debug(f"服装匹配分数: {clothing_score:.3f}")
        
        # 位置匹配（较低权重）
        if target_features['position']:
            position_weight = 0.1
            total_weight += position_weight
            
            # 位置匹配需要结合检测框的位置信息
            position_score = 0.5  # 给予基础分数
            score += position_score * position_weight
            logger.debug(f"位置匹配分数: {position_score:.3f}")
        
        # 归一化分数
        if total_weight > 0:
            score = score / total_weight
        
        return min(score, 1.0)
    
    def find_matching_persons(self, detections: List[Dict], 
                            description: str, 
                            min_score: float = 0.1) -> List[Tuple[Dict, float]]:
        """
        在检测结果中查找匹配描述的人物
        
        Args:
            detections (List[Dict]): 人物检测结果
            description (str): 文本描述
            min_score (float): 最小匹配分数阈值
            
        Returns:
            List[Tuple[Dict, float]]: 匹配的人物和分数列表
        """
        # 解析目标描述
        target_features = self.parse_description(description)
        
        matches = []
        
        for detection in detections:
            # 提取人物的颜色特征
            person_crop = detection.get('person_crop')
            if person_crop is not None and person_crop.size > 0:
                color_features = self.extract_color_features(person_crop)
                
                person_features = {
                    'colors': color_features,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence']
                }
                
                # 计算匹配分数
                match_score = self.calculate_match_score(person_features, target_features)
                
                # 添加调试信息
                logger.info(f"检测到的颜色特征: {color_features}")
                logger.info(f"匹配分数: {match_score:.3f} (阈值: {min_score})")
                
                if match_score >= min_score:
                    matches.append((detection, match_score))
                    logger.info(f"找到匹配: 分数={match_score:.3f}, 位置={detection['bbox']}")
                else:
                    logger.info(f"未达到阈值: 分数={match_score:.3f} < {min_score}")
        
        # 按分数降序排序
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches

if __name__ == "__main__":
    # 测试代码
    matcher = TextMatcher()
    
    # 测试描述解析
    test_descriptions = [
        "a man with red helmet",
        "红色头盔的人",
        "蓝色上衣的女士",
        "左边的人"
    ]
    
    print("测试文本描述解析:")
    for desc in test_descriptions:
        features = matcher.parse_description(desc)
        print(f"  '{desc}' -> {features}") 