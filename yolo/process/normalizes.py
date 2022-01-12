import torch

__all__ = ["normalize"]

# NANODET_NORM = [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
NANODET_NORM = [[123.675, 116.28, 103.53], [58.395, 57.12, 57.375]]


def _normalize(img, mean, std):
    """
    Args:
        img (torch.Tensor): N, C, H, W
    """
    mean = (
        torch.tensor(mean, dtype=img.dtype, device=img.device).reshape(1, 3, 1, 1)
        / 255.0
    )
    std = (
        torch.tensor(std, dtype=img.dtype, device=img.device).reshape(1, 3, 1, 1)
        / 255.0
    )
    img = (img - mean) / std
    return img


def yolov5_norm(images):
    """
    Args:
        images (torch.Tensor): N, C, H, W
    """
    return images / 255.0


def yolox_norm(images):
    """
    Args:
        images (torch.Tensor): N, C, H, W
    """
    return images


def nanodet_norm(images):
    """
    Args:
        images (torch.Tensor): N, C, H, W
    """
    images = images / 255.0
    images = _normalize(images, *NANODET_NORM)
    return images


normalize = {
    "yolov5": yolov5_norm,
    "yolox": yolox_norm,
    "nanodet": nanodet_norm,
    "yolo-fastestv2": yolov5_norm,
    "yolov5-lite": yolov5_norm,
}
