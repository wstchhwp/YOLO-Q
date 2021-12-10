from .inference_utils import letterbox, non_max_suppression, scale_coords
from .utils import plot_one_box
import os
import cv2
import numpy as np
from itertools import repeat
import torch


class Predictor(object):
    """docstring for Predictor."""

    def __init__(self, img_hw, models, model_type='yolov5'):
        super(Predictor, self).__init__()
        self.img_hw = img_hw
        self.ori_hw = []
        self.models = models
        # TODO
        self.device = 0
        self.half = 0
        self.multi_model = True if isinstance(models, list) else False

        assert model_type in ['yolov5', 'yolox']
        self.normlize = model_type is not 'yolox'

    def preprocess_one_img(self, image, auto=True):
        """Preprocess one image"""
        if type(image) == str and os.path.isfile(image):
            img_raw = cv2.imread(image)
        else:
            img_raw = image
        resized_img, _, _ = letterbox(img_raw, new_shape=self.img_hw, auto=auto)
        # cv2.imshow('x', resized_img)
        # cv2.waitKey(0)
        # H, W, C -> 1, C, H, W
        resized_img = resized_img[:, :, ::-1].transpose(2, 0, 1)[None, :]
        resized_img = np.ascontiguousarray(resized_img)
        self.ori_hw.append(img_raw.shape[:2])
        return resized_img

    def preprocess_multi_img(self, images, auto=True):
        """Preprocess multi image"""
        resized_imgs = []
        for image in images:
            img = self.preprocess_one_img(image, auto=auto)
            resized_imgs.append(img)
        imgs = np.stack(resized_imgs, axis=0)
        return imgs

    def inference(self, images):
        """Inference.
        
        Args:
            images (numpy.ndarray | list[numpy.ndarray]): Input images.
            # images (torch.Tensor): B, C, H, W.
        """
        if isinstance(images, list):
            imgs = self.preprocess_one_img(images)
        else:
            imgs = self.preprocess_multi_img(images)
        imgs = torch.from_numpy(imgs).to(self.device)
        imgs = imgs.half() if self.half else imgs.float()  # uint8 to fp16/32
        if self.normlize:
            imgs = imgs / 255.

        if self.multi_model:
            return self.inference_multi_model(imgs)
        else:
            return self.inference_single_model(imgs)
    
    def inference_single_model(self, images):
        preds = self.model(images)
        if self.normalize:
            preds = preds[0]
        outputs = self.postprocess(preds)
        # B, num_boxes, classes+5
        return outputs

    def inference_multi_model(self, images):
        pass

    def postprocess(self, preds, conf_thres=0.4, iou_thres=0.5, classes=None):
        """Postprocess multi images

        Args:
            preds (torch.Tensor): [B, num_boxes, classes+5].
        """
        outputs = non_max_suppression(
            preds, conf_thres, iou_thres, classes=classes, agnostic=False
        )
        for i, det in enumerate(outputs):  # detections per image
            if det is None or len(det) == 0:
                continue
            det[:, :4] = scale_coords(
                self.img_hw, det[:, :4], self.ori_hw[i]
            ).round()
        return outputs

    def visualize_one_img(self, img, output, vis_conf=0.4):
        """Visualize one images
        
        Args:
            imgs (numpy.array): one images.
            outputs (torch.Tensor): one outputs.
            vis_confs (float, optional): Visualize threshold.
        """
        if output is None or len(output) == 0:
            return
        for *xyxy, conf, cls in reversed(output[:, :6]):
            if conf < vis_conf:
                continue
            # label = '%s %.2f' % (self.names[int(cls)], conf)
            label = '%s' % (self.names[int(cls)])
            color = self.colors[int(cls)]
            plot_one_box(xyxy, img, label=label,
                         color=color, 
                         line_thickness=2)
        return img

    def visualize_multi_img(self, imgs, outputs, vis_confs=0.4):
        """Visualize multi images
        
        Args:
            imgs (list[numpy.array]): multi images.
            outputs (torch.Tensor): multi outputs.
            vis_confs (float | tuple[float], optional): Visualize threshold.
        """
        if isinstance(vis_confs, float):
            vis_confs = list(repeat(vis_confs, len(imgs)))
        assert len(imgs) == len(outputs) == len(vis_confs)
        for i, output in enumerate(outputs):  # detections per image
            self.visualize_one_img(imgs[i], output, vis_confs[i])
        return imgs
