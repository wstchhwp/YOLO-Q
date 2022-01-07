from .detectors.yolov5 import Model
from .detectors.yolox import YOLOX
from .layers.yolo_head import YOLOXHead
from .layers.yolo_pafpn import YOLOPAFPN
from ..utils.torch_utils import select_device
import torch
import torch.nn as nn
from omegaconf import OmegaConf

yolox_type = {
    "nano": {"depth": 0.33, "width": 0.25, "depthwise": True,},
    "tiny": {"depth": 0.33, "width": 0.375, "depthwise": False,},
    "s": {"depth": 0.33, "width": 0.50, "depthwise": False,},
    "m": {"depth": 0.67, "width": 0.75, "depthwise": False,},
    "l": {"depth": 1.0, "width": 1.0, "depthwise": False,},
    "x": {"depth": 1.33, "width": 1.25, "depthwise": False,},
}

def build_yolov5(cfg, weight_path, device, half=True):
    device = select_device(device)
    # TODO, torch.no_grad() support inference only.
    # TODO, device may move to `Predictor`
    with torch.no_grad():
        model = Model(cfg).to(device)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt, strict=False)
        model.fuse().eval()
        if half:
            model.half()
    return model

def build_yolox(cfg, weight_path, device, half=True):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    with open(cfg, 'r') as f:
        cfg = OmegaConf.load(f)
    model_cfg = yolox_type[cfg.type]
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(model_cfg['depth'], model_cfg['width'], in_channels=in_channels, depthwise=model_cfg['depthwise'])
    head = YOLOXHead(cfg.nc, model_cfg['width'], in_channels=in_channels, depthwise=model_cfg['depthwise'])
    model = YOLOX(backbone, head)
    setattr(model, 'names', cfg.names)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)

    # load checkpoint
    # TODO, torch.no_grad() support inference only.
    # TODO, device may move to `Predictor`
    device = select_device(device)
    with torch.no_grad():
        model.to(device).eval()
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if half:
            model.half()
    return model

def build_from_configs(cfg_path):
    with open(cfg_path, 'r') as f:
        # config = yaml.safe_load(f)
        config =OmegaConf.load(f)

    model_list = []
    for _, v in config.items():
        assert v.model_type in ['yolov5', 'yolox']
        if v.model_type == 'yolov5':
            builder = build_yolov5
        else:
            builder = build_yolox
        model = builder(cfg=v.yaml,
                        weight_path=v.weight,
                        device='0')
        setattr(model, 'model_type', v.model_type)
        setattr(model, 'conf_thres', v.conf_thres)
        setattr(model, 'iou_thres', v.iou_thres)
        setattr(model, 'filter', v.filter)
        model_list.append(model)
    if len(config) > 1:
        return model_list
    else:
        return model_list[0]
