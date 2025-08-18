#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人体检测模块
使用YOLO模型检测视频中的人物
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonDetector:
    """人体检测器类"""
    
    def __init__(self, model_name='weights/yolov8n.pt'):
        """
        初始化人体检测器
        
        Args:
            model_name (str): YOLO模型名称，默认使用YOLOv8n（较小但快速的模型）
        """
        try:
            # 加载YOLO模型，处理PyTorch 2.6+的权重加载限制
            import torch
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            # 设置PyTorch允许不安全的权重加载
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.nn.modules.head.Detect',
                    'ultralytics.nn.modules.block.C2f',
                    'ultralytics.nn.modules.conv.Conv',
                    'ultralytics.nn.modules.block.SPPF',
                    'ultralytics.nn.modules.block.Bottleneck'
                ])
            
            # 临时修补torch.load以允许不安全权重
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return original_load(*args, **kwargs)
            torch.load = patched_load
            
            try:
                self.model = YOLO(model_name)
                logger.info(f"成功加载YOLO模型: {model_name}")
            finally:
                # 恢复原始的torch.load
                torch.load = original_load
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
            
        # COCO数据集中人的类别ID是0
        self.person_class_id = 0
        
    def detect_persons_in_frame(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        在单帧图像中检测人物
        
        Args:
            frame (np.ndarray): 输入图像帧
            confidence_threshold (float): 置信度阈值
            
        Returns:
            List[Dict]: 检测到的人物信息列表
        """
        detections = []
        
        try:
            # 使用YOLO模型进行检测
            results = self.model(frame, verbose=False)
            
            # 处理检测结果
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # 获取类别ID和置信度
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # 只处理人类检测结果且置信度高于阈值
                        if class_id == self.person_class_id and confidence >= confidence_threshold:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # 提取人物区域
                            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                            
                            detection_info = {
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence,
                                'person_crop': person_crop,
                                'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
                            }
                            
                            detections.append(detection_info)
                            
        except Exception as e:
            logger.error(f"检测过程中出现错误: {e}")
            
        return detections
    
    def detect_persons_in_video(self, video_path: str, max_frames: int = 100) -> List[Tuple[int, List[Dict]]]:
        """
        在视频中检测人物
        
        Args:
            video_path (str): 视频文件路径
            max_frames (int): 最大处理帧数
            
        Returns:
            List[Tuple[int, List[Dict]]]: (帧号, 检测结果)的列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return []
            
        frame_detections = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"开始处理视频: {video_path}")
        logger.info(f"总帧数: {total_frames}, 计划处理: {min(max_frames, total_frames)} 帧")
        
        # 计算跳帧间隔，确保均匀采样
        skip_frames = max(1, total_frames // max_frames) if max_frames < total_frames else 1
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 跳帧处理
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % skip_frames != 0:
                continue
                
            # 检测当前帧中的人物
            detections = self.detect_persons_in_frame(frame)
            
            if detections:  # 只保存有检测结果的帧
                frame_detections.append((current_frame_number, detections))
                logger.info(f"帧 {current_frame_number}: 检测到 {len(detections)} 个人物")
                
            frame_count += 1
            
            # 显示进度
            if frame_count % 10 == 0:
                progress = (frame_count / max_frames) * 100
                logger.info(f"处理进度: {progress:.1f}%")
                
        cap.release()
        logger.info(f"视频处理完成，共检测到 {len(frame_detections)} 帧包含人物")
        
        return frame_detections

if __name__ == "__main__":
    # 测试代码
    detector = PersonDetector()
    
    # 测试视频文件路径
    video_path = "data/DJI_20250807124730_0002_S.MP4"
    
    print("正在测试人体检测...")
    detections = detector.detect_persons_in_video(video_path, max_frames=50)
    
    print(f"检测完成！共在 {len(detections)} 帧中发现人物。")
    for frame_num, frame_detections in detections[:5]:  # 显示前5个结果
        print(f"  帧 {frame_num}: {len(frame_detections)} 个人物") 