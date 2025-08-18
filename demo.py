#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频人物查找系统演示脚本
展示系统工作流程而不需要实际的视频处理
"""

import numpy as np
import cv2
from text_matcher import TextMatcher
from colorama import Fore, Style, init

# 初始化colorama
init(autoreset=True)

def create_demo_person_image(width=200, height=300, colors=['red', 'blue']):
    """创建一个模拟的人物图像用于演示"""
    # 创建空白图像
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 根据指定颜色填充不同区域
    color_map = {
        'red': (0, 0, 255),      # BGR format
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }
    
    # 头部区域 (头盔)
    if 'red' in colors:
        cv2.rectangle(image, (50, 20), (150, 80), color_map['red'], -1)
    
    # 身体区域 (上衣)
    if 'blue' in colors:
        cv2.rectangle(image, (40, 80), (160, 220), color_map['blue'], -1)
    
    # 腿部区域
    if len(colors) > 2 and colors[2] in color_map:
        cv2.rectangle(image, (45, 220), (155, 290), color_map[colors[2]], -1)
    else:
        cv2.rectangle(image, (45, 220), (155, 290), color_map['black'], -1)
    
    return image

def demo_text_parsing():
    """演示文本解析功能"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}演示 1: 文本描述解析")
    print(f"{Fore.CYAN}{'='*60}")
    
    matcher = TextMatcher()
    
    # 测试不同的描述
    test_descriptions = [
        "a man with red helmet",
        "红色头盔的人", 
        "blue shirt person",
        "蓝色上衣的人",
        "person with red helmet and blue shirt",
        "戴红色头盔穿蓝色上衣的人",
        "left side person",
        "左边的人"
    ]
    
    print(f"{Fore.YELLOW}正在解析以下文本描述：\n")
    
    for i, description in enumerate(test_descriptions, 1):
        features = matcher.parse_description(description)
        print(f"{Fore.WHITE}{i}. 描述: \"{description}\"")
        print(f"   {Fore.GREEN}颜色: {features['colors']}")
        print(f"   {Fore.GREEN}服装: {features['clothing']}")
        print(f"   {Fore.GREEN}位置: {features['position']}")
        print()

def demo_color_extraction():
    """演示颜色特征提取功能"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}演示 2: 颜色特征提取")
    print(f"{Fore.CYAN}{'='*60}")
    
    matcher = TextMatcher()
    
    # 创建不同颜色组合的测试图像
    test_cases = [
        (['red'], "红色头盔人物"),
        (['red', 'blue'], "红色头盔+蓝色上衣人物"),
        (['blue', 'green'], "蓝色上衣+绿色裤子人物"),
        (['red', 'blue', 'yellow'], "红色头盔+蓝色上衣+黄色裤子人物")
    ]
    
    print(f"{Fore.YELLOW}正在分析不同颜色组合的人物图像：\n")
    
    for i, (colors, description) in enumerate(test_cases, 1):
        # 创建模拟图像
        test_image = create_demo_person_image(colors=colors)
        
        # 提取颜色特征
        color_features = matcher.extract_color_features(test_image)
        
        print(f"{Fore.WHITE}{i}. {description}")
        print(f"   {Fore.GREEN}图像颜色: {colors}")
        print(f"   {Fore.GREEN}检测到的颜色特征:")
        for color, ratio in color_features.items():
            print(f"     - {color}: {ratio:.2%}")
        print()

def demo_matching():
    """演示匹配功能"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}演示 3: 人物匹配")
    print(f"{Fore.CYAN}{'='*60}")
    
    matcher = TextMatcher()
    
    # 创建几个模拟的人物检测结果
    simulated_detections = []
    
    # 人物1: 红色头盔 + 蓝色上衣
    person1_image = create_demo_person_image(colors=['red', 'blue'])
    detection1 = {
        'bbox': (100, 50, 300, 350),
        'confidence': 0.9,
        'person_crop': person1_image,
        'center': (200, 200)
    }
    simulated_detections.append(detection1)
    
    # 人物2: 蓝色头盔 + 绿色上衣
    person2_image = create_demo_person_image(colors=['blue', 'green'])
    detection2 = {
        'bbox': (400, 50, 600, 350),
        'confidence': 0.8,
        'person_crop': person2_image,
        'center': (500, 200)
    }
    simulated_detections.append(detection2)
    
    # 人物3: 红色头盔 + 黄色上衣
    person3_image = create_demo_person_image(colors=['red', 'yellow'])
    detection3 = {
        'bbox': (700, 50, 900, 350),
        'confidence': 0.85,
        'person_crop': person3_image,
        'center': (800, 200)
    }
    simulated_detections.append(detection3)
    
    # 测试不同的查找描述
    test_queries = [
        "a man with red helmet",
        "蓝色上衣的人",
        "red helmet and blue shirt",
        "person with yellow clothing"
    ]
    
    print(f"{Fore.YELLOW}模拟场景: 检测到3个人物")
    print(f"  - 人物1: 红色头盔 + 蓝色上衣")
    print(f"  - 人物2: 蓝色头盔 + 绿色上衣") 
    print(f"  - 人物3: 红色头盔 + 黄色上衣")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"{Fore.WHITE}{i}. 查找: \"{query}\"")
        
        # 查找匹配的人物
        matches = matcher.find_matching_persons(simulated_detections, query, min_score=0.1)
        
        if matches:
            print(f"   {Fore.GREEN}找到 {len(matches)} 个匹配:")
            for j, (detection, score) in enumerate(matches):
                person_idx = simulated_detections.index(detection) + 1
                print(f"     - 人物{person_idx}: 匹配分数 {score:.3f}")
        else:
            print(f"   {Fore.RED}未找到匹配的人物")
        print()

def demo_workflow():
    """演示完整工作流程"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}演示 4: 完整工作流程")
    print(f"{Fore.CYAN}{'='*60}")
    
    print(f"{Fore.YELLOW}模拟视频人物查找的完整流程：")
    print()
    
    # 步骤1
    print(f"{Fore.WHITE}步骤 1: 输入文本描述")
    description = "a man with red helmet"
    print(f"   用户输入: \"{description}\"")
    print()
    
    # 步骤2
    print(f"{Fore.WHITE}步骤 2: 解析文本描述")
    matcher = TextMatcher()
    features = matcher.parse_description(description)
    print(f"   解析结果:")
    print(f"   - 颜色: {features['colors']}")
    print(f"   - 服装: {features['clothing']}")
    print()
    
    # 步骤3
    print(f"{Fore.WHITE}步骤 3: 模拟视频帧处理")
    print(f"   正在处理视频帧...")
    print(f"   - 帧 100: 检测到 2 个人物")
    print(f"   - 帧 250: 检测到 1 个人物") 
    print(f"   - 帧 400: 检测到 3 个人物")
    print()
    
    # 步骤4
    print(f"{Fore.WHITE}步骤 4: 特征匹配")
    # 创建匹配的人物图像
    matching_person = create_demo_person_image(colors=['red', 'blue'])
    color_features = matcher.extract_color_features(matching_person)
    print(f"   分析人物颜色特征:")
    for color, ratio in color_features.items():
        print(f"   - {color}: {ratio:.2%}")
    print()
    
    # 步骤5
    print(f"{Fore.WHITE}步骤 5: 生成结果")
    print(f"   {Fore.GREEN}✓ 找到匹配的人物!")
    print(f"   - 最佳匹配: 帧 100, 分数 0.847")
    print(f"   - 第二匹配: 帧 400, 分数 0.726")
    print(f"   - 结果已保存到 results/ 目录")
    print()

def main():
    """主演示函数"""
    print(f"{Fore.CYAN}🎥 视频人物查找系统 - 交互式演示")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}这个演示展示了系统如何使用文本描述查找视频中的特定人物")
    print(f"{Fore.YELLOW}由于演示环境限制，我们使用模拟数据来展示核心功能")
    
    # 运行各个演示
    demo_text_parsing()
    demo_color_extraction()
    demo_matching()
    demo_workflow()
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}演示完成!")
    print(f"{Fore.YELLOW}要在真实视频上测试系统，请运行:")
    print(f"{Fore.WHITE}python video_person_finder.py <视频文件> \"<人物描述>\"")
    print(f"{Fore.CYAN}{'='*60}")

if __name__ == "__main__":
    main() 