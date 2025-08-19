#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频片段分析器
对短视频片段进行逐帧详细分析，展示检测与语义理解过程
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from argparse import ArgumentParser
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from drone_optimized_detector import DroneOptimizedPersonFinder
from utils.clip_utils import load_clip, compute_clip_similarity
from utils.dino_utils import load_dino, compute_dino_features, compute_dino_similarity


def create_analysis_overlay(frame: np.ndarray, detections: List[Dict], 
                          clip_scores: List[float], frame_num: int, 
                          query_text: str) -> np.ndarray:
    """创建详细分析覆盖图"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # 改进的标题信息（右上角，半透明）
    title_bg_height = 140
    title_width = 500
    # 完全不透明的黑色背景，确保文字清晰
    cv2.rectangle(overlay, (w - title_width, 0), (w, title_bg_height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (w - title_width, 0), (w, title_bg_height), (255, 255, 255), 2)
    
    # 处理中文显示（简化为英文避免编码问题）
    display_query = query_text
    if '红色头盔' in query_text:
        display_query = 'red helmet'
    elif '黄色' in query_text:
        display_query = 'yellow vest'
    elif '蓝色' in query_text:
        display_query = 'blue clothing'
    
    title_lines = [
        f"Frame {frame_num}",
        f"Query: {display_query}",
        f"Detected: {len(detections)} persons",
        f"Max CLIP: {max(clip_scores):.3f}" if clip_scores else "Max CLIP: N/A"
    ]
    
    for i, line in enumerate(title_lines):
        cv2.putText(overlay, line, (w - title_width + 15, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 先绘制所有检测框（细线，浅色）
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 180, 180), 2)
        cv2.putText(overlay, f"P{i+1}", (x1+2, max(y1-8, 20)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    
    # 再绘制高CLIP分数的检测框（粗线，鲜艳色）
    for i, (det, clip_score) in enumerate(zip(detections, clip_scores)):
        if clip_score < 0.20:  # 适应新的分数范围
            continue
            
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # 根据CLIP语义分数选择颜色
        if clip_score > 0.70:
            color = (0, 0, 255)  # 红色 - 高匹配
            thickness = 6
        elif clip_score > 0.50:
            color = (0, 255, 255)  # 黄色 - 中匹配
            thickness = 5
        elif clip_score > 0.30:
            color = (0, 150, 255)  # 橙色 - 中低匹配
            thickness = 4
        else:
            color = (0, 255, 0)  # 绿色 - 低匹配
            thickness = 3
        
        # 绘制检测框
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        
        # 大字体清晰标签
        main_label = f"P{i+1} CLIP:{clip_score:.3f}"
        font_scale = 0.8
        thickness_text = 2
        
        # 标签位置（避免重叠，优先上方）
        label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)[0]
        label_x = max(x1, 10)
        label_y = max(y1 - 35, 40)
        
        # 白色背景 + 黑色边框，确保最佳对比度
        cv2.rectangle(overlay, (label_x - 5, label_y - 25), 
                     (label_x + label_size[0] + 10, label_y + 8), (255, 255, 255), -1)
        cv2.rectangle(overlay, (label_x - 5, label_y - 25), 
                     (label_x + label_size[0] + 10, label_y + 8), (0, 0, 0), 2)
        
        # 黑色文字，最大对比度
        cv2.putText(overlay, main_label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_text)
        
        # 如果是最高分，添加特殊标记
        if clip_scores and clip_score == max(clip_scores) and clip_score > 0.25:
            center_x, center_y = (x1+x2)//2, (y1+y2)//2
            cv2.circle(overlay, (center_x, center_y), 25, (0, 0, 255), 4)
            # TARGET标签用白底黑字
            target_label = "TARGET"
            target_size = cv2.getTextSize(target_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            cv2.rectangle(overlay, (center_x - target_size[0]//2 - 8, center_y + 30), 
                         (center_x + target_size[0]//2 + 8, center_y + 65), (255, 255, 255), -1)
            cv2.rectangle(overlay, (center_x - target_size[0]//2 - 8, center_y + 30), 
                         (center_x + target_size[0]//2 + 8, center_y + 65), (0, 0, 0), 2)
            cv2.putText(overlay, target_label, (center_x - target_size[0]//2, center_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    
    return overlay


def create_crop_analysis_grid(detections: List[Dict], clip_scores: List[float], 
                            query_text: str, frame_num: int, output_dir: str):
    """创建人物裁剪分析网格图"""
    if not detections:
        return
    
    # 计算网格布局
    n_persons = len(detections)
    cols = min(4, n_persons)
    rows = (n_persons + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'Frame {frame_num} - Person Crops Analysis\nQuery: "{query_text}"', 
                fontsize=14, fontweight='bold')
    
    for i, (det, clip_score) in enumerate(zip(detections, clip_scores)):
        if i >= len(axes):
            break
            
        crop = det.get('person_crop')
        if crop is not None and crop.size > 0:
            # 转换BGR到RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            axes[i].imshow(crop_rgb)
            
            # 标题信息
            conf = det['confidence']
            size = det['size']
            title = f'Person {i+1}\nCLIP: {clip_score:.3f}\nConf: {conf:.3f}\nSize: {size[0]}×{size[1]}'
            
            # 根据CLIP分数设置标题颜色
            if clip_score > 0.6:
                title_color = 'red'
            elif clip_score > 0.4:
                title_color = 'orange'
            else:
                title_color = 'green'
                
            axes[i].set_title(title, fontsize=10, color=title_color, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'No Crop', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Person {i+1}\nNo Crop Available', fontsize=10)
        
        axes[i].axis('off')
    
    # 隐藏未使用的子图
    for i in range(n_persons, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'frame_{frame_num}_crops_analysis.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()


def analyze_video_segment(video_path: str, query_text: str, 
                         start_frame: int = 0, num_frames: int = 10,
                         finder: DroneOptimizedPersonFinder = None,
                         use_dino: bool = False,
                         reference_sample: np.ndarray = None) -> Dict[str, Any]:
    """分析视频片段"""
    
    start_time = time.time()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backend_name = 'dino' if use_dino else 'clip'
    output_dir = os.path.join('outputs', f'segment_analysis_{backend_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📹 开始分析视频片段...")
    print(f"   视频: {video_path}")
    print(f"   查询: {query_text}")
    print(f"   语义后端: {'DINO' if use_dino else 'CLIP'}")
    print(f"   帧范围: {start_frame} - {start_frame + num_frames}")
    print(f"   输出: {output_dir}")
    
    # 使用传入的检测器或创建默认检测器
    if finder is None:
        finder = DroneOptimizedPersonFinder()
    
    # 加载语义理解模型
    clip_load_start = time.time()
    semantic_ctx = None
    reference_features = None
    
    if use_dino:
        semantic_ctx = load_dino('dinov2')
        if semantic_ctx is None:
            print("⚠️ DINOv2未加载，将使用基础检测分数")
        else:
            clip_load_time = time.time() - clip_load_start
            print(f"✅ DINOv2加载完成 ({clip_load_time:.2f}秒)")
            # 如果提供了参考样本，提取参考特征
            if reference_sample is not None:
                reference_features = compute_dino_features(semantic_ctx, reference_sample)
                print(f"✅ 参考样本特征提取完成")
    else:
        semantic_ctx = load_clip()
        if semantic_ctx is None:
            print("⚠️ CLIP未加载，将使用基础检测分数")
        else:
            clip_load_time = time.time() - clip_load_start
            print(f"✅ CLIP加载完成 ({clip_load_time:.2f}秒)")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return {}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 视频信息: {total_frames}帧, {fps:.1f}FPS")
    
    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    analysis_results = {
        'video_info': {
            'path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'start_frame': start_frame,
            'num_frames': num_frames
        },
        'query_text': query_text,
        'semantic_backend': 'DINO' if use_dino else 'CLIP',
        'frame_analyses': [],
        'summary': {},
        'output_dir': output_dir,
        'timing': {
            'start_time': start_time,
            'semantic_load_time': clip_load_time if semantic_ctx else 0.0
        }
    }
    
    all_detections = []
    all_semantic_scores = []
    detection_time = 0.0
    semantic_time = 0.0
    
    # 逐帧分析
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame = start_frame + i
        print(f"\n🔍 分析帧 {current_frame}...")
        
        # 检测人物
        det_start = time.time()
        detections = finder.detect_all_persons_in_frame(frame)
        detection_time += time.time() - det_start
        print(f"   检测到 {len(detections)} 个人物")
        
        # 语义分析
        semantic_start = time.time()
        semantic_scores = []
        
        if use_dino and semantic_ctx:
            # DINOv2视觉相似度分析
            if reference_features is not None:
                for det in detections:
                    crop = det.get('person_crop')
                    score = compute_dino_similarity(semantic_ctx, crop, reference_features)
                    semantic_scores.append(score if score is not None else 0.0)
            else:
                semantic_scores = [0.0] * len(detections)
                
        elif not use_dino and semantic_ctx and query_text:
            # CLIP文本语义分析
            # 更多样化的提示扩展，增大分数差异
            prompts = [query_text]
            text_lower = query_text.lower()
            
            # 通用扩展
            prompts.extend([
                f"a person {query_text}",
                f"person with {query_text}",
                f"person wearing {query_text}"
            ])
            
            # 负面提示（用于对比，降低不匹配的分数）
            negative_prompts = [
                "person without the described features",
                "different person",
                "unrelated person"
            ]
            
            # 针对性扩展
            if 'green' in text_lower and ('shirt' in text_lower or 't-shirt' in text_lower):
                prompts.extend([
                    "green t-shirt",
                    "person in green shirt", 
                    "green clothing",
                    "person wearing green top",
                    "green casual wear"
                ])
                negative_prompts.extend([
                    "person not wearing green",
                    "person in different colored clothing",
                    "red clothing", "blue clothing"
                ])
            elif 'red' in text_lower and 'helmet' in text_lower:
                prompts.extend([
                    "red safety helmet",
                    "construction worker with red helmet",
                    "red hard hat",
                    "person with red protective gear"
                ])
                negative_prompts.extend([
                    "person without helmet",
                    "person with different colored helmet",
                    "blue helmet", "yellow helmet", "white helmet"
                ])
            elif 'white' in text_lower and ('shirt' in text_lower or 't-shirt' in text_lower):
                prompts.extend([
                    "white t-shirt",
                    "person in white shirt",
                    "white clothing",
                    "white casual wear"
                ])
                negative_prompts.extend([
                    "person not wearing white",
                    "colored clothing",
                    "dark clothing"
                ])
             
            for det in detections:
                crop = det.get('person_crop')
                
                # 计算正面提示的最大相似度
                max_score = 0.0
                for prompt in prompts:
                    score = compute_clip_similarity(semantic_ctx, crop, prompt)
                    if score is not None:
                        max_score = max(max_score, score)
                
                # 计算负面提示的最大相似度
                max_negative_score = 0.0
                for neg_prompt in negative_prompts:
                    neg_score = compute_clip_similarity(semantic_ctx, crop, neg_prompt)
                    if neg_score is not None:
                        max_negative_score = max(max_negative_score, neg_score)
                
                # 基础增强分数
                enhanced_score = max_score - max_negative_score * 0.5
                semantic_scores.append(enhanced_score)
            
            # 后处理：对整帧的分数进行百分位拉伸
            if semantic_scores:
                scores_array = np.array(semantic_scores)
                
                # 计算百分位数
                p10 = np.percentile(scores_array, 10)
                p90 = np.percentile(scores_array, 90)
                
                # 拉伸到0-1范围，同时保持相对关系
                stretched_scores = []
                for score in semantic_scores:
                    if p90 > p10:
                        # 线性拉伸到0-1
                        normalized = (score - p10) / (p90 - p10)
                        stretched = max(0.0, min(1.0, normalized))
                        
                        # 进一步非线性增强对比度
                        if stretched > 0.7:
                            final = 0.7 + 0.3 * ((stretched - 0.7) / 0.3) ** 0.5
                        elif stretched < 0.3:
                            final = 0.3 * (stretched / 0.3) ** 1.5
                        else:
                            final = stretched
                    else:
                        final = 0.5  # 所有分数相同时的默认值
                    
                    stretched_scores.append(max(0.0, min(1.0, final)))
                
                semantic_scores = stretched_scores
        else:
            semantic_scores = [0.0] * len(detections)
            
        semantic_time += time.time() - semantic_start
        
        # 创建分析覆盖图
        analysis_overlay = create_analysis_overlay(frame, detections, semantic_scores, 
                                                 current_frame, query_text)
        
        # 保存帧分析结果
        cv2.imwrite(os.path.join(output_dir, f'frame_{current_frame:04d}_analysis.jpg'), 
                   analysis_overlay)
        
        # 创建裁剪分析网格
        create_crop_analysis_grid(detections, semantic_scores, query_text, 
                                current_frame, output_dir)
        
        # 记录分析数据
        frame_analysis = {
            'frame_number': current_frame,
            'detection_count': len(detections),
            'detections': [],
            'max_semantic_score': max(semantic_scores) if semantic_scores else 0.0,
            'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
        }
        
        for j, (det, semantic_score) in enumerate(zip(detections, semantic_scores)):
            det_info = {
                'person_id': j + 1,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'size': det['size'],
                'semantic_score': semantic_score,
                'center': det['center']
            }
            frame_analysis['detections'].append(det_info)
        
        analysis_results['frame_analyses'].append(frame_analysis)
        all_detections.extend(detections)
        all_semantic_scores.extend(semantic_scores)
    
    cap.release()
    
    total_time = time.time() - start_time
    
    # 生成汇总分析
    if all_detections:
        analysis_results['summary'] = {
            'total_detections': len(all_detections),
            'frames_with_persons': len([f for f in analysis_results['frame_analyses'] if f['detection_count'] > 0]),
            'avg_persons_per_frame': np.mean([f['detection_count'] for f in analysis_results['frame_analyses']]),
            'max_semantic_score': max(all_semantic_scores) if all_semantic_scores else 0.0,
            'avg_semantic_score': np.mean(all_semantic_scores) if all_semantic_scores else 0.0,
            'avg_confidence': np.mean([d['confidence'] for d in all_detections])
        }
    
    # 添加性能统计
    analysis_results['timing'].update({
        'total_time': total_time,
        'detection_time': detection_time,
        'semantic_time': semantic_time,
        'avg_time_per_frame': total_time / max(1, num_frames),
        'detection_fps': num_frames / max(detection_time, 0.001),
        'semantic_fps': len(all_detections) / max(semantic_time, 0.001)
    })
    
    # 保存JSON报告
    with open(os.path.join(output_dir, 'segment_analysis_report.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    # 创建汇总可视化
    create_summary_visualization(analysis_results, output_dir)
    
    print(f"\n✅ 分析完成！")
    print(f"⏱️  总耗时: {total_time:.2f}秒")
    print(f"🔍 检测耗时: {detection_time:.2f}秒 ({num_frames/max(detection_time,0.001):.1f} FPS)")
    backend_name = 'DINO' if use_dino else 'CLIP'
    print(f"🧠 {backend_name}耗时: {semantic_time:.2f}秒 ({len(all_detections)/max(semantic_time,0.001):.1f} crops/sec)")
    print(f"📁 结果保存在: {output_dir}")
    print(f"📊 总检测: {len(all_detections)} 个人物")
    print(f"🎯 最高{backend_name}分数: {max(all_semantic_scores):.3f}" if all_semantic_scores else "")
    print(f"📈 {backend_name}分数范围: {min(all_semantic_scores):.3f} - {max(all_semantic_scores):.3f}" if all_semantic_scores else "")
    
    return analysis_results


def create_summary_visualization(analysis_results: Dict, output_dir: str):
    """创建汇总可视化图表"""
    frame_analyses = analysis_results['frame_analyses']
    if not frame_analyses:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Video Segment Analysis Summary\nQuery: "{analysis_results["query_text"]}"', 
                fontsize=14, fontweight='bold')
    
    # 1. 每帧检测数量
    frame_nums = [f['frame_number'] for f in frame_analyses]
    detection_counts = [f['detection_count'] for f in frame_analyses]
    
    ax1.bar(range(len(frame_nums)), detection_counts, alpha=0.7, color='skyblue')
    ax1.set_title('Persons Detected per Frame')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Number of Persons')
    ax1.set_xticks(range(len(frame_nums)))
    ax1.set_xticklabels([f'F{f}' for f in frame_nums], rotation=45)
    
    # 2. CLIP分数分布
    all_semantic_scores = []
    for f in frame_analyses:
        for det in f['detections']:
            all_semantic_scores.append(det['semantic_score'])
    
    if all_semantic_scores:
        ax2.hist(all_semantic_scores, bins=20, alpha=0.7, color='lightcoral')
        ax2.axvline(np.mean(all_semantic_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_semantic_scores):.3f}')
        ax2.set_title('Semantic Score Distribution')
        ax2.set_xlabel('Semantic Score')
        ax2.set_ylabel('Count')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No Semantic Scores', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Semantic Score Distribution')
    
    # 3. 检测置信度分布
    all_confidences = []
    for f in frame_analyses:
        for det in f['detections']:
            all_confidences.append(det['confidence'])
    
    if all_confidences:
        ax3.hist(all_confidences, bins=20, alpha=0.7, color='lightgreen')
        ax3.axvline(np.mean(all_confidences), color='green', linestyle='--',
                   label=f'Mean: {np.mean(all_confidences):.3f}')
        ax3.set_title('Detection Confidence Distribution')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Count')
        ax3.legend()
    
    # 4. 人物尺寸分布
    all_sizes = []
    for f in frame_analyses:
        for det in f['detections']:
            size = det['size']
            all_sizes.append(size[0] * size[1])  # 面积
    
    if all_sizes:
        ax4.hist(all_sizes, bins=20, alpha=0.7, color='gold')
        ax4.axvline(np.mean(all_sizes), color='orange', linestyle='--',
                   label=f'Mean: {np.mean(all_sizes):.0f}')
        ax4.set_title('Person Size Distribution (Area)')
        ax4.set_xlabel('Area (pixels²)')
        ax4.set_ylabel('Count')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_analysis.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()


def create_timeline_view(analysis_results: Dict, output_dir: str):
    """创建时间轴视图"""
    frame_analyses = analysis_results['frame_analyses']
    if not frame_analyses:
        return
    
    # 创建时间轴图像
    timeline_height = 200
    frame_width = 150
    total_width = len(frame_analyses) * frame_width
    
    timeline = np.ones((timeline_height, total_width, 3), dtype=np.uint8) * 255
    
    for i, frame_analysis in enumerate(frame_analyses):
        x_start = i * frame_width
        x_end = (i + 1) * frame_width
        
        # 绘制帧信息
        frame_num = frame_analysis['frame_number']
        detection_count = frame_analysis['detection_count']
        max_clip = frame_analysis['max_semantic_score'] # Changed from max_clip_score to max_semantic_score
        
        # 根据检测数量选择背景颜色
        if detection_count == 0:
            bg_color = (240, 240, 240)  # 灰色
        elif detection_count <= 2:
            bg_color = (200, 255, 200)  # 浅绿
        elif detection_count <= 4:
            bg_color = (255, 255, 200)  # 浅黄
        else:
            bg_color = (255, 200, 200)  # 浅红
        
        cv2.rectangle(timeline, (x_start, 0), (x_end-5, timeline_height), bg_color, -1)
        cv2.rectangle(timeline, (x_start, 0), (x_end-5, timeline_height), (100, 100, 100), 2)
        
        # 添加文字信息
        texts = [
            f'F{frame_num}',
            f'{detection_count}P',
            f'C:{max_clip:.2f}' if max_clip > 0 else 'C:N/A'
        ]
        
        for j, text in enumerate(texts):
            y_pos = 30 + j * 40
            cv2.putText(timeline, text, (x_start + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.imwrite(os.path.join(output_dir, 'timeline_view.jpg'), timeline)


def create_analysis_video(analysis_results: Dict, output_dir: str, fps: float = 5.0):
    """创建分析过程视频"""
    frame_analyses = analysis_results['frame_analyses']
    if not frame_analyses:
        return
    
    print(f"🎬 生成分析视频...")
    
    # 获取第一帧尺寸
    first_frame_path = os.path.join(output_dir, f'frame_{frame_analyses[0]["frame_number"]:04d}_analysis.jpg')
    if not os.path.exists(first_frame_path):
        print("❌ 找不到分析帧图像")
        return
    
    sample_frame = cv2.imread(first_frame_path)
    if sample_frame is None:
        print("❌ 无法读取分析帧图像")
        return
    
    height, width = sample_frame.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, 'analysis_video.mp4')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ 无法创建视频文件")
        return
    
    # 写入每一帧
    for frame_analysis in frame_analyses:
        frame_num = frame_analysis['frame_number']
        frame_path = os.path.join(output_dir, f'frame_{frame_num:04d}_analysis.jpg')
        
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                # 调整尺寸确保一致
                frame = cv2.resize(frame, (width, height))
                out.write(frame)
                print(f"   添加帧 {frame_num}")
            else:
                print(f"⚠️ 无法读取帧 {frame_num}")
        else:
            print(f"⚠️ 找不到帧文件 {frame_path}")
    
    out.release()
    
    print(f"✅ 视频生成完成: {video_path}")
    print(f"📹 视频参数: {len(frame_analyses)}帧, {fps}FPS, {width}x{height}")


def main():
    ap = ArgumentParser(description='Analyze video segment with frame-by-frame detection and semantic understanding')
    ap.add_argument('video', help='video path')
    ap.add_argument('query', help='query text, e.g., 红色头盔')
    ap.add_argument('--start-frame', type=int, default=1000, help='starting frame number')
    ap.add_argument('--num-frames', type=int, default=30, help='number of frames to analyze')
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--iou', type=float, default=0.35)
    ap.add_argument('--min-size', type=int, default=6)
    ap.add_argument('--tile', type=int, default=960)
    ap.add_argument('--overlap', type=float, default=0.45)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--tta', action='store_true', help='enable TTA for higher recall')
    ap.add_argument('--progressive', action='store_true', help='enable progressive detection for max recall')
    ap.add_argument('--weights', type=str, default='weights/yolov8m.pt', help='YOLO weights')
    ap.add_argument('--output-video', action='store_true', help='generate output video with analysis overlays')
    ap.add_argument('--fps', type=float, default=10.0, help='output video FPS')
    ap.add_argument('--use-dino', action='store_true', help='use DINOv2 instead of CLIP for visual similarity')
    ap.add_argument('--reference-sample', type=str, default='', help='reference image for DINO visual similarity')
    
    args = ap.parse_args()
    
    # 处理参考样本
    reference_sample = None
    if args.use_dino and args.reference_sample:
        try:
            reference_sample = cv2.imread(args.reference_sample)
            if reference_sample is not None:
                print(f"✅ 加载参考样本: {args.reference_sample}")
            else:
                print(f"⚠️ 无法加载参考样本: {args.reference_sample}")
        except Exception as e:
            print(f"⚠️ 参考样本加载错误: {e}")
    
    # 构建检测器参数
    detector_kwargs = {
        'model_name': args.weights,
        'min_person_size': args.min_size,
        'confidence_threshold': args.conf,
        'nms_threshold': args.iou,
        'tile_size': args.tile,
        'tile_overlap': args.overlap,
        'tile_imgsz': args.imgsz
    }
    
    # 设置检测器增强选项
    finder = DroneOptimizedPersonFinder(**detector_kwargs)
    setattr(finder, 'use_tta', bool(args.tta))
    setattr(finder, 'use_progressive', bool(args.progressive))
    
    # 运行分析
    results = analyze_video_segment(
        args.video, args.query, 
        args.start_frame, args.num_frames,
        finder=finder,
        use_dino=args.use_dino,
        reference_sample=reference_sample
    )
    
    # 创建时间轴视图
    if results:
        output_dir = results.get('output_dir', 'outputs/segment_analysis_latest')
        create_timeline_view(results, output_dir)
        
        # 生成输出视频
        if args.output_video:
            create_analysis_video(results, output_dir, args.fps)
        
        print(f"\n📋 生成的文件:")
        print(f"   - frame_XXXX_analysis.jpg: 逐帧详细分析")
        print(f"   - frame_XXXX_crops_analysis.png: 人物裁剪网格")
        print(f"   - summary_analysis.png: 汇总统计图表")
        print(f"   - timeline_view.jpg: 时间轴视图")
        print(f"   - segment_analysis_report.json: 详细数据报告")
        if args.output_video:
            print(f"   - analysis_video.mp4: 分析过程视频")


if __name__ == '__main__':
    main() 