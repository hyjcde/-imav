#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频人物查找主程序
使用文本描述在视频中查找特定人物
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import argparse
import logging
from typing import List, Tuple, Dict, Any
from colorama import Fore, Back, Style, init

# 导入自定义模块
from person_detector import PersonDetector
from text_matcher import TextMatcher

# 初始化colorama用于彩色输出
init(autoreset=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoPersonFinder:
    """视频人物查找器"""
    
    def __init__(self):
        """初始化查找器"""
        try:
            self.detector = PersonDetector()
            self.matcher = TextMatcher()
            logger.info("视频人物查找器初始化成功")
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def find_person_in_video(self, video_path: str, description: str, 
                           max_frames: int = 100, output_dir: str = "outputs/results") -> Dict[str, Any]:
        """
        在视频中查找匹配描述的人物
        
        Args:
            video_path (str): 视频文件路径
            description (str): 目标人物描述
            max_frames (int): 最大处理帧数
            output_dir (str): 结果输出目录
            
        Returns:
            Dict[str, Any]: 查找结果
        """
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}开始在视频中查找人物")
        print(f"{Fore.CYAN}视频文件: {video_path}")
        print(f"{Fore.CYAN}描述: {description}")
        print(f"{Fore.CYAN}{'='*60}\n")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 第一步：检测视频中的所有人物
        print(f"{Fore.YELLOW}步骤 1: 检测视频中的人物...")
        frame_detections = self.detector.detect_persons_in_video(video_path, max_frames)
        
        if not frame_detections:
            print(f"{Fore.RED}未在视频中检测到任何人物！")
            return {"status": "no_persons_detected", "matches": []}
        
        print(f"{Fore.GREEN}检测完成！共在 {len(frame_detections)} 帧中发现人物")
        
        # 第二步：在检测结果中查找匹配的人物
        print(f"\n{Fore.YELLOW}步骤 2: 查找匹配描述的人物...")
        
        all_matches = []
        cap = cv2.VideoCapture(video_path)
        
        for frame_idx, (frame_num, detections) in enumerate(frame_detections):
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # 查找匹配的人物
            matches = self.matcher.find_matching_persons(detections, description)
            
            if matches:
                for detection, score in matches:
                    match_info = {
                        'frame_number': frame_num,
                        'detection': detection,
                        'match_score': score,
                        'frame': frame.copy()
                    }
                    all_matches.append(match_info)
                    
                print(f"{Fore.GREEN}帧 {frame_num}: 找到 {len(matches)} 个匹配项")
        
        cap.release()
        
        # 第三步：生成结果
        if all_matches:
            print(f"\n{Fore.GREEN}查找成功！共找到 {len(all_matches)} 个匹配项")
            
            # 按匹配分数排序
            all_matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            # 显示最佳匹配结果
            print(f"\n{Fore.CYAN}最佳匹配结果：")
            for i, match in enumerate(all_matches[:5]):  # 显示前5个最佳匹配
                print(f"  {i+1}. 帧 {match['frame_number']}: 分数 {match['match_score']:.3f}")
            
            # 保存结果图像
            self._save_results(all_matches, description, output_dir)
            
            return {
                "status": "success",
                "matches": all_matches,
                "total_matches": len(all_matches),
                "output_dir": output_dir
            }
        else:
            print(f"{Fore.RED}未找到匹配描述的人物！")
            return {"status": "no_matches", "matches": []}
    
    def _save_results(self, matches: List[Dict], description: str, output_dir: str):
        """
        保存查找结果到图像文件
        
        Args:
            matches (List[Dict]): 匹配结果列表
            description (str): 描述文本
            output_dir (str): 输出目录
        """
        print(f"\n{Fore.YELLOW}正在保存结果图像...")
        
        # 保存前5个最佳匹配
        for i, match in enumerate(matches[:5]):
            frame = match['frame']
            detection = match['detection']
            score = match['match_score']
            frame_num = match['frame_number']
            
            # 在图像上绘制检测框
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 添加文本标签
            label = f"Match: {score:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 保存图像
            output_filename = f"match_{i+1}_frame_{frame_num}_score_{score:.3f}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            
            print(f"  保存: {output_filename}")
        
        # 创建结果摘要图
        self._create_summary_plot(matches[:9], description, output_dir)
    
    def _create_summary_plot(self, matches: List[Dict], description: str, output_dir: str):
        """
        创建结果摘要图
        
        Args:
            matches (List[Dict]): 匹配结果列表
            description (str): 描述文本
            output_dir (str): 输出目录
        """
        if not matches:
            return
            
        # 创建3x3子图网格
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Person Search Results: "{description}"', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, match in enumerate(matches):
            if i >= 9:  # 最多显示9个结果
                break
                
            frame = match['frame']
            detection = match['detection']
            score = match['match_score']
            frame_num = match['frame_number']
            
            # 提取人物区域
            x1, y1, x2, y2 = detection['bbox']
            person_crop = frame[y1:y2, x1:x2]
            
            # 转换颜色格式 (BGR -> RGB)
            if person_crop.size > 0:
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                axes[i].imshow(person_crop_rgb)
            
            axes[i].set_title(f'Frame {frame_num}\nScore: {score:.3f}', 
                            fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # 隐藏未使用的子图
        for i in range(len(matches), 9):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # 保存摘要图
        summary_path = os.path.join(output_dir, "search_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  保存摘要图: search_summary.png")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='在视频中查找特定人物')
    parser.add_argument('video', help='视频文件路径')
    parser.add_argument('description', help='人物描述')
    parser.add_argument('--max-frames', type=int, default=100, 
                       help='最大处理帧数 (默认: 100)')
    parser.add_argument('--output-dir', default='outputs/results', 
                       help='输出目录 (默认: outputs/results)')
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"{Fore.RED}错误: 视频文件不存在: {args.video}")
        return
    
    try:
        # 创建查找器并开始查找
        finder = VideoPersonFinder()
        results = finder.find_person_in_video(
            args.video, 
            args.description, 
            args.max_frames, 
            args.output_dir
        )
        
        # 显示最终结果
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}查找完成！")
        
        if results['status'] == 'success':
            print(f"{Fore.GREEN}状态: 成功")
            print(f"{Fore.GREEN}找到匹配项: {results['total_matches']}")
            print(f"{Fore.GREEN}结果保存在: {results['output_dir']}")
        else:
            print(f"{Fore.YELLOW}状态: {results['status']}")
        
        print(f"{Fore.CYAN}{'='*60}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"{Fore.RED}错误: {e}")

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认测试参数
    import sys
    if len(sys.argv) == 1:
        print(f"{Fore.CYAN}使用默认测试参数...")
        test_video = "data/DJI_20250807124730_0002_S.MP4"
        test_description = "a man with red helmet"
        
        if os.path.exists(test_video):
            try:
                finder = VideoPersonFinder()
                results = finder.find_person_in_video(test_video, test_description, max_frames=50)
            except Exception as e:
                print(f"{Fore.RED}测试失败: {e}")
        else:
            print(f"{Fore.RED}测试视频文件不存在: {test_video}")
            print("\n使用方法:")
            print("python video_person_finder.py <视频文件> <人物描述>")
            print('例如: python video_person_finder.py video.mp4 "a man with red helmet"')
    else:
        main() 