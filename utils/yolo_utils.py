#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any


def safe_predict(model: Any, image: Any,
                 conf: float = None,
                 iou: float = None,
                 imgsz: int = None,
                 verbose: bool = False):
    """
    兼容不同 ultralytics 版本的安全预测调用。
    优先尝试带阈值/尺寸的调用，失败则逐步退化到最小参数调用。
    """
    # 1. 直接调用 __call__ 带全部参数
    try:
        kwargs = {'verbose': verbose}
        if conf is not None:
            kwargs['conf'] = conf
        if iou is not None:
            kwargs['iou'] = iou
        if imgsz is not None:
            kwargs['imgsz'] = imgsz
        return model(image, **kwargs)
    except TypeError:
        pass
    except Exception:
        pass

    # 2. 显式调用 predict 带全部参数
    try:
        kwargs = {'verbose': verbose}
        if conf is not None:
            kwargs['conf'] = conf
        if iou is not None:
            kwargs['iou'] = iou
        if imgsz is not None:
            kwargs['imgsz'] = imgsz
        return model.predict(image, **kwargs)
    except TypeError:
        pass
    except Exception:
        pass

    # 3. 仅 verbose（去掉不兼容的阈值/尺寸参数）
    try:
        return model(image, verbose=verbose)
    except Exception:
        pass

    # 4. 最后尝试 predict 最小参数
    return model.predict(image)  # 若仍失败，让上层捕获异常 