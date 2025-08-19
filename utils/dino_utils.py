#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, List
import numpy as np
from PIL import Image
import torch


def load_dino(model_type: str = 'dinov2') -> Optional[Dict[str, Any]]:
    """
    加载DINO模型。支持DINOv2 (自监督特征) 和 DINO-DETR (目标检测)
    
    Args:
        model_type: 'dinov2' 或 'dino_detr'
        
    Returns:
        DINO上下文字典或None
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'dinov2':
        try:
            # DINOv2 自监督视觉特征
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            model = model.to(device)
            model.eval()
            
            # 简单的图像预处理
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            return {
                'backend': 'dinov2',
                'model': model,
                'preprocess': preprocess,
                'device': device,
                'feature_dim': 384  # DINOv2-ViT-S特征维度
            }
        except Exception as e:
            print(f"DINOv2加载失败: {e}")
    
    elif model_type == 'dino_detr':
        try:
            # DINO-DETR 目标检测
            from transformers import DetrImageProcessor, DetrForObjectDetection
            
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            model = model.to(device)
            model.eval()
            
            return {
                'backend': 'dino_detr',
                'model': model,
                'processor': processor,
                'device': device
            }
        except Exception as e:
            print(f"DINO-DETR加载失败: {e}")
    
    return None


def compute_dino_features(dino_ctx: Dict[str, Any], image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    使用DINOv2提取图像特征向量
    
    Args:
        dino_ctx: DINO上下文
        image_bgr: BGR格式图像
        
    Returns:
        特征向量 (384维) 或 None
    """
    if dino_ctx is None or dino_ctx['backend'] != 'dinov2':
        return None
        
    if image_bgr is None or image_bgr.size == 0:
        return None
    
    try:
        # BGR -> RGB PIL
        image_rgb = Image.fromarray(image_bgr[:, :, ::-1])
        
        # 预处理
        image_tensor = dino_ctx['preprocess'](image_rgb).unsqueeze(0).to(dino_ctx['device'])
        
        # 提取特征
        with torch.no_grad():
            features = dino_ctx['model'](image_tensor)
            # DINOv2返回的是CLS token特征
            features = features.cpu().numpy().flatten()
        
        return features
        
    except Exception as e:
        print(f"DINOv2特征提取失败: {e}")
        return None


def compute_dino_similarity(dino_ctx: Dict[str, Any], image_bgr: np.ndarray, 
                           reference_features: np.ndarray) -> Optional[float]:
    """
    计算图像与参考特征的DINOv2相似度
    
    Args:
        dino_ctx: DINO上下文
        image_bgr: 待匹配图像
        reference_features: 参考特征向量
        
    Returns:
        余弦相似度 [0,1] 或 None
    """
    if reference_features is None:
        return None
        
    current_features = compute_dino_features(dino_ctx, image_bgr)
    if current_features is None:
        return None
    
    try:
        # 余弦相似度计算
        dot_product = np.dot(current_features, reference_features)
        norm_current = np.linalg.norm(current_features)
        norm_reference = np.linalg.norm(reference_features)
        
        if norm_current == 0 or norm_reference == 0:
            return 0.0
            
        cosine_sim = dot_product / (norm_current * norm_reference)
        
        # 映射到 [0,1]
        return max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))
        
    except Exception:
        return None


def create_reference_features_from_query(dino_ctx: Dict[str, Any], 
                                       query_samples: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    从查询样本图像创建参考特征（平均特征向量）
    
    Args:
        dino_ctx: DINO上下文
        query_samples: 查询样本图像列表
        
    Returns:
        平均特征向量或None
    """
    if not query_samples or dino_ctx is None:
        return None
    
    features_list = []
    for sample in query_samples:
        features = compute_dino_features(dino_ctx, sample)
        if features is not None:
            features_list.append(features)
    
    if not features_list:
        return None
    
    # 计算平均特征
    mean_features = np.mean(features_list, axis=0)
    return mean_features


def detect_objects_with_dino_detr(dino_ctx: Dict[str, Any], 
                                 image_bgr: np.ndarray) -> List[Dict]:
    """
    使用DINO-DETR进行目标检测
    
    Args:
        dino_ctx: DINO-DETR上下文
        image_bgr: 输入图像
        
    Returns:
        检测结果列表
    """
    if dino_ctx is None or dino_ctx['backend'] != 'dino_detr':
        return []
        
    if image_bgr is None or image_bgr.size == 0:
        return []
    
    try:
        # BGR -> RGB PIL
        image_rgb = Image.fromarray(image_bgr[:, :, ::-1])
        
        # 预处理
        inputs = dino_ctx['processor'](images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(dino_ctx['device']) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = dino_ctx['model'](**inputs)
        
        # 后处理
        target_sizes = torch.tensor([image_rgb.size[::-1]]).to(dino_ctx['device'])
        results = dino_ctx['processor'].post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 1:  # COCO person class
                x1, y1, x2, y2 = box.cpu().numpy()
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(score),
                    'class_id': int(label),
                    'source': 'dino_detr'
                })
        
        return detections
        
    except Exception as e:
        print(f"DINO-DETR检测失败: {e}")
        return []


def compute_visual_similarity_score(dino_ctx: Dict[str, Any], 
                                   crop1: np.ndarray, crop2: np.ndarray) -> Optional[float]:
    """
    计算两个图像裁剪的DINOv2视觉相似度
    用于基于样本的相似图像检索
    
    Args:
        dino_ctx: DINOv2上下文
        crop1, crop2: 两个图像裁剪
        
    Returns:
        相似度分数 [0,1] 或 None
    """
    features1 = compute_dino_features(dino_ctx, crop1)
    features2 = compute_dino_features(dino_ctx, crop2)
    
    if features1 is None or features2 is None:
        return None
    
    # 余弦相似度
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (norm1 * norm2)
    return max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0)) 