#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的颜色分析模块
专门针对无人机俯视角度的头盔和服装颜色检测
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class EnhancedColorAnalyzer:
    """增强的颜色分析器，针对无人机俯视角度优化"""
    
    def __init__(self):
        """初始化颜色分析器"""
        # 针对俯视角度优化的HSV颜色范围
        self.hsv_ranges = {
            # 红色（头盔常见颜色）
            'red_1': [(0, 120, 100), (8, 255, 255)],
            'red_2': [(172, 120, 100), (180, 255, 255)],
            
            # 其他常见头盔颜色
            'orange': [(8, 120, 100), (20, 255, 255)],      # 橙色
            'yellow': [(20, 120, 100), (30, 255, 255)],     # 黄色
            'blue': [(100, 120, 100), (130, 255, 255)],     # 蓝色
            'green': [(40, 120, 100), (80, 255, 255)],      # 绿色
            'white': [(0, 0, 200), (180, 55, 255)],         # 白色
            'black': [(0, 0, 0), (180, 255, 50)],           # 黑色
            'gray': [(0, 0, 50), (180, 55, 200)],           # 灰色
        }
        
        # 颜色权重（头部区域权重更高）
        self.region_weights = {
            'head': 1.5,    # 头部区域（上1/3）
            'torso': 1.0,   # 躯干区域（中1/3）
            'legs': 0.7     # 腿部区域（下1/3）
        }
    
    def analyze_person_colors(self, person_crop: np.ndarray) -> Dict[str, float]:
        """
        分析人物的颜色特征，重点关注头部区域
        
        Args:
            person_crop: 人物裁剪图像
            
        Returns:
            Dict: 颜色特征字典
        """
        if person_crop is None or person_crop.size == 0:
            return {}
        
        height, width = person_crop.shape[:2]
        
        # 将人物区域分为3个部分
        head_region = person_crop[:height//3, :]      # 头部（上1/3）
        torso_region = person_crop[height//3:2*height//3, :]  # 躯干（中1/3）
        legs_region = person_crop[2*height//3:, :]    # 腿部（下1/3）
        
        regions = {
            'head': head_region,
            'torso': torso_region, 
            'legs': legs_region
        }
        
        color_features = {}
        
        # 分别分析每个区域的颜色
        for region_name, region_img in regions.items():
            if region_img.size > 0:
                region_colors = self._extract_colors_from_region(region_img)
                weight = self.region_weights[region_name]
                
                # 应用权重
                for color, ratio in region_colors.items():
                    feature_name = f"{region_name}_{color}"
                    color_features[feature_name] = ratio * weight
                    
                    # 同时记录整体颜色（加权平均）
                    if color in color_features:
                        color_features[color] += ratio * weight
                    else:
                        color_features[color] = ratio * weight
        
        # 归一化整体颜色值
        total_weight = sum(self.region_weights.values())
        for color in list(color_features.keys()):
            if '_' not in color:  # 只处理整体颜色，不处理区域特定颜色
                color_features[color] /= total_weight
        
        return color_features
    
    def _extract_colors_from_region(self, region: np.ndarray) -> Dict[str, float]:
        """从图像区域提取颜色特征"""
        if region is None or region.size == 0:
            return {}
        
        # 转换为HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        total_pixels = region.shape[0] * region.shape[1]
        
        color_ratios = {}
        
        for color_name, (lower, upper) in self.hsv_ranges.items():
            # 创建颜色掩码
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # 特殊处理红色（需要两个范围）
            if color_name == 'red_1':
                lower2, upper2 = self.hsv_ranges['red_2']
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask, mask2)
                color_name = 'red'  # 合并为红色
            elif color_name == 'red_2':
                continue  # 跳过red_2，已经在red_1中处理
            
            # 计算颜色占比
            color_pixels = cv2.countNonZero(mask)
            ratio = color_pixels / total_pixels
            
            # 只保留有意义的颜色占比
            if ratio > 0.05:  # 至少占5%
                color_ratios[color_name] = ratio
        
        return color_ratios
    
    def find_helmet_candidates(self, persons: List[Dict]) -> List[Dict]:
        """
        找到可能戴头盔的人物候选
        
        Args:
            persons: 人物检测结果列表
            
        Returns:
            List: 按头盔可能性排序的候选列表
        """
        helmet_candidates = []
        
        for person in persons:
            person_crop = person.get('person_crop')
            if person_crop is None or person_crop.size == 0:
                continue
            
            # 分析颜色特征
            color_features = self.analyze_person_colors(person_crop)
            
            # 计算头盔评分
            helmet_score = self._calculate_helmet_score(color_features)
            
            candidate = {
                'person': person,
                'color_features': color_features,
                'helmet_score': helmet_score,
                'head_colors': {k: v for k, v in color_features.items() if k.startswith('head_')}
            }
            
            helmet_candidates.append(candidate)
        
        # 按头盔评分排序
        helmet_candidates.sort(key=lambda x: x['helmet_score'], reverse=True)
        
        return helmet_candidates
    
    def _calculate_helmet_score(self, color_features: Dict[str, float]) -> float:
        """
        计算头盔可能性评分
        
        Args:
            color_features: 颜色特征字典
            
        Returns:
            float: 头盔评分 (0-1)
        """
        score = 0.0
        
        # 头部区域有明显颜色的加分
        head_colors = {k: v for k, v in color_features.items() if k.startswith('head_')}
        
        # 头盔常见颜色权重
        helmet_color_weights = {
            'red': 1.0,
            'orange': 1.0,
            'yellow': 1.0,
            'blue': 0.8,
            'white': 0.6,
            'green': 0.5
        }
        
        for feature_name, ratio in head_colors.items():
            color = feature_name.replace('head_', '')
            if color in helmet_color_weights:
                weight = helmet_color_weights[color]
                score += ratio * weight
        
        # 头部区域颜色集中度加分
        if len(head_colors) > 0:
            max_head_color_ratio = max(head_colors.values())
            score += max_head_color_ratio * 0.5  # 颜色集中度奖励
        
        return min(score, 1.0)  # 限制在[0,1]范围内
    
    def visualize_color_analysis(self, person_crop: np.ndarray, color_features: Dict[str, float], 
                               output_path: str = None):
        """
        可视化颜色分析结果
        
        Args:
            person_crop: 人物裁剪图像
            color_features: 颜色特征
            output_path: 输出路径
        """
        if person_crop is None or person_crop.size == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(person_rgb)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 分割区域
        height = person_crop.shape[0]
        head_region = person_crop[:height//3, :]
        torso_region = person_crop[height//3:2*height//3, :]
        legs_region = person_crop[2*height//3:, :]
        
        regions = [
            (head_region, '头部区域', (0, 1)),
            (torso_region, '躯干区域', (0, 2)),
            (legs_region, '腿部区域', (1, 0))
        ]
        
        for region, title, pos in regions:
            if region.size > 0:
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                axes[pos].imshow(region_rgb)
                axes[pos].set_title(title)
                axes[pos].axis('off')
        
        # 颜色分析结果
        head_colors = {k.replace('head_', ''): v for k, v in color_features.items() if k.startswith('head_')}
        overall_colors = {k: v for k, v in color_features.items() if '_' not in k}
        
        # 头部颜色分布
        if head_colors:
            colors = list(head_colors.keys())
            values = list(head_colors.values())
            axes[1, 1].bar(colors, values, alpha=0.7)
            axes[1, 1].set_title('头部颜色分布')
            axes[1, 1].set_ylabel('占比')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 整体颜色分布
        if overall_colors:
            colors = list(overall_colors.keys())
            values = list(overall_colors.values())
            axes[1, 2].bar(colors, values, alpha=0.7, color='orange')
            axes[1, 2].set_title('整体颜色分布')
            axes[1, 2].set_ylabel('占比')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def test_color_analysis():
    """测试颜色分析功能"""
    analyzer = EnhancedColorAnalyzer()
    
    # 测试数据
    test_image = np.zeros((90, 60, 3), dtype=np.uint8)
    # 模拟红色头盔
    test_image[:30, :] = [0, 0, 255]  # 红色头部
    test_image[30:60, :] = [255, 255, 255]  # 白色躯干
    test_image[60:, :] = [0, 0, 0]  # 黑色腿部
    
    # 分析颜色
    colors = analyzer.analyze_person_colors(test_image)
    
    print("测试颜色分析结果:")
    for color, ratio in colors.items():
        print(f"  {color}: {ratio:.3f}")
    
    # 可视化
    analyzer.visualize_color_analysis(test_image, colors, "test_color_analysis.png")

if __name__ == "__main__":
    test_color_analysis() 