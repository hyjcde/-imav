#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional
import numpy as np
from PIL import Image


def load_clip() -> Optional[Dict[str, Any]]:
    """
    Try to load CLIP. Prefer OpenAI CLIP; fallback to OpenCLIP.
    Returns a dict with: { 'backend': 'clip'|'open_clip', 'model', 'preprocess', 'device', 'tokenize' }
    If both backends unavailable, return None.
    """
    device = 'cpu'
    # Try OpenAI CLIP
    try:
        import torch
        import clip  # type: ignore
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)
        return {
            'backend': 'clip',
            'model': model,
            'preprocess': preprocess,
            'device': device,
            'tokenize': clip.tokenize,
        }
    except Exception:
        pass

    # Fallback: OpenCLIP
    try:
        import torch
        import open_clip  # type: ignore
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        model = model.to(device)
        return {
            'backend': 'open_clip',
            'model': model,
            'preprocess': preprocess,
            'device': device,
            'tokenize': open_clip.tokenize,
        }
    except Exception:
        return None


def compute_clip_similarity(clip_ctx: Dict[str, Any], image_bgr: np.ndarray, text: str) -> Optional[float]:
    """
    Compute cosine similarity between image crop (BGR numpy) and text using CLIP.
    Returns a float in [0, 1] (cos mapped from [-1,1] to [0,1]).
    """
    if clip_ctx is None or image_bgr is None or image_bgr.size == 0 or not text:
        return None
    try:
        import torch
        device = clip_ctx['device']
        # BGR -> RGB PIL
        image_rgb = Image.fromarray(image_bgr[:, :, ::-1])
        image_tensor = clip_ctx['preprocess'](image_rgb).unsqueeze(0).to(device)
        text_tokens = clip_ctx['tokenize']([text]).to(device)
        with torch.no_grad():
            if clip_ctx['backend'] == 'clip':
                image_feat = clip_ctx['model'].encode_image(image_tensor)
                text_feat = clip_ctx['model'].encode_text(text_tokens)
            else:
                image_feat = clip_ctx['model'].encode_image(image_tensor)
                text_feat = clip_ctx['model'].encode_text(text_tokens)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            sim = (image_feat @ text_feat.T).squeeze().item()
        # map [-1,1] -> [0,1]
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))
    except Exception:
        return None 