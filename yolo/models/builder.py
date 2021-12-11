from .detectors.yolov5 import Model
from .detectors.yolox import YOLOX
from .layers.yolo_head import YOLOXHead
from .layers.yolo_pafpn import YOLOPAFPN
from ..utils.torch_utils import select_device
from ..utils.plots import plot_one_box
from ..data.datasets import letterbox
from ..utils.boxes import non_max_suppression, scale_coords
import torch
import torch.nn as nn
import random
import cv2
import os
import numpy as np
from yolox.exp import get_exp

yolox_type = {
    "nano": {"depth": 0.33, "width": 0.25, "depthwise": True,},
    "tiny": {"depth": 0.33, "width": 0.375, "depthwise": False,},
    "s": {"depth": 0.33, "width": 0.50, "depthwise": False,},
    "m": {"depth": 0.67, "width": 0.75, "depthwise": False,},
    "l": {"depth": 1.0, "width": 1.0, "depthwise": False,},
    "x": {"depth": 1.33, "width": 1.25, "depthwise": False,},
}

def build_yolov5(cfg, weight_path, device):
    device = select_device(device)
    # TODO
    with torch.no_grad():
        model = Model(cfg).to(device)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt, strict=False)
        model.fuse().eval()
    return model

def build_yolox(model_type, weight_path, device, num_classes):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    cfg = yolox_type[model_type]
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(cfg['depth'], cfg['width'], in_channels=in_channels, depthwise=cfg['depthwise'])
    head = YOLOXHead(num_classes, cfg['width'], in_channels=in_channels, depthwise=cfg['depthwise'])
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)

    # load checkpoint
    # TODO
    device = select_device(device)
    with torch.no_grad():
        model.to(device).eval()
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    return model

class Yolov5:
    def __init__(self, cfg, weight_path, device, img_hw=(384, 640)):
        self.weights = weight_path
        self.device = select_device(device)
        self.half = True
        # path aware
        with torch.no_grad():
            self.model = Model(cfg).to(self.device)
            checkpoint = torch.load(weight_path)
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.fuse().eval()

        if self.half:
            self.model.half()  # to FP16
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        self.colors = [
            [random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))
        ]
        self.show = False
        self.img_hw = img_hw
        self.pause = False

    def preprocess(self, image, auto=True):  # (h, w)
        if type(image) == str and os.path.isfile(image):
            img0 = cv2.imread(image)
        else:
            img0 = image
        # img, _, _ = letterbox(img0, new_shape=new_shape)
        img, _, _ = letterbox(img0, new_shape=self.img_hw, auto=auto)
        # cv2.imshow('x', img)
        # cv2.waitKey(0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, img0

    def dynamic_detect(
        self,
        image,
        img0s,
        areas=None,
        classes=None,
        conf_threshold=0.7,
        iou_threshold=0.4,
    ):
        output = {}
        if classes is not None:
            for c in classes:
                output[self.names[int(c)]] = 0
        else:
            for n in self.names:
                output[n] = 0
        img = torch.from_numpy(image).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        torch.cuda.synchronize()
        pred = self.model(img)[0]
        pred = non_max_suppression(
            pred, conf_threshold, iou_threshold, classes=classes, agnostic=False
        )

        torch.cuda.synchronize()
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0s[i].shape
                ).round()
                # if areas is not None and len(areas[i]):
                #     _, warn = polygon_ROIarea(
                #         det[:, :4], areas[i], img0s[i])
                #     det = det[warn]
                #     pred[i] = det
                for di, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    output[self.names[int(cls)]] += 1
                    # label = '%s %.2f' % (self.names[int(cls)], conf)
                    label = "%s" % (self.names[int(cls)])
                    # label = None
                    # color = [0, 0, 255] if conf < 0.6 else self.colors[int(cls)]
                    color = self.colors[int(cls)]
                    # if self.show:
                    plot_one_box(
                        xyxy, img0s[i], label=label, color=color, line_thickness=2
                    )

        if self.show:
            for i in range(len(img0s)):
                cv2.namedWindow(f"p{i}", cv2.WINDOW_NORMAL)
                cv2.imshow(f"p{i}", img0s[i])
            key = cv2.waitKey(0 if self.pause else 1)
            self.pause = True if key == ord(" ") else False
            if key == ord("q") or key == ord("e") or key == 27:
                exit()
        return pred, output
