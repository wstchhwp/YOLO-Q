from .yolov5.models.yolo import Model
from .utils import select_device
import torch
from yolox.exp import get_exp


def build_yolov5(cfg, weight_path, device):
    device = select_device(device)
    with torch.no_grad():
        model = Model(cfg).to(device)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt, strict=False)
        model.fuse().eval()
    return model

def build_yolox(exp_path, weight_path, device):
    device = select_device(device)
    exp = get_exp(exp_path)
    with torch.no_grad():
        model = exp.get_model()
        model.to(device).eval()
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    return model

