#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
颜色检测调优脚本
用于优化HSV颜色范围，提高颜色识别准确性
"""

import cv2
import numpy as np

def improved_hsv_ranges():
    """改进的HSV颜色范围，更精确的颜色检测"""
    return {
        # 红色需要两个范围（HSV色轮的两端）
        'red_1': [(0, 100, 100), (10, 255, 255)],     # 红色范围1
        'red_2': [(160, 100, 100), (180, 255, 255)],  # 红色范围2
        
        # 其他颜色优化范围
        'blue': [(100, 100, 50), (130, 255, 255)],    # 蓝色
        'green': [(40, 40, 40), (80, 255, 255)],      # 绿色
        'yellow': [(20, 100, 100), (30, 255, 255)],   # 黄色
        'orange': [(10, 100, 100), (20, 255, 255)],   # 橙色
        'purple': [(130, 100, 50), (160, 255, 255)],  # 紫色
        
        # 灰度颜色优化
        'white': [(0, 0, 200), (180, 55, 255)],       # 白色
        'black': [(0, 0, 0), (180, 255, 50)],         # 黑色  
        'gray': [(0, 0, 50), (180, 55, 200)],         # 灰色
    }

def test_color_detection():
    """测试颜色检测效果"""
    print("改进的颜色检测范围:")
    ranges = improved_hsv_ranges()
    for color, (lower, upper) in ranges.items():
        print(f"  {color}: {lower} - {upper}")

if __name__ == "__main__":
    test_color_detection() 