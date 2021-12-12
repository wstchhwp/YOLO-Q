from ..data.datasets import letterbox
from ..utils.boxes import non_max_suppression, scale_coords
from ..utils.plots import plot_one_box
from ..utils.general import to_2tuple
from ..utils.torch_utils import select_device
import os
import cv2
import numpy as np
from itertools import repeat
import torch


class Predictor(object):
    """Predictor for multi models and multi images.

    Args:
        img_hw (int | tuple[int]): Input size.
        models (nn.Module | List[nn.Module]): Models, support different model in a list.
        device (str): Device.
        model_type (str | List[str]): Model type, cause `yolov5` has same different operation
            (like div 255.), example: 'yolox' or ['yolov5', 'yolox'].
        half (bool, optional): Whether use fp16 to inference.
    """

    def __init__(self, img_hw, models, device, half=True):
        super(Predictor, self).__init__()
        img_hw = to_2tuple(img_hw) if isinstance(img_hw, int) else img_hw

        self.img_hw = img_hw
        self.ori_hw = []
        self.models = models
        self.device = select_device(device)
        self.half = half
        self.multi_model = True if isinstance(models, list) else False

        # self._is_yolov5(model_type)

    def _is_yolov5(self, model_type):
        if self.multi_model:
            if isinstance(model_type, str):
                model_type = list(repeat(model_type, len(self.models)))
            assert len(self.models) == len(model_type)
            # TODO
            for m in model_type:
                assert m in ["yolov5", "yolox"]
            self.yolov5 = [m is not "yolox" for m in model_type]
        else:
            assert model_type in ["yolov5", "yolox"]
            self.yolov5 = model_type is not "yolox"

    def preprocess_one_img(self, image, auto=True, center_padding=True):
        """Preprocess one image.

        Args:
            image (numpy.ndarray | str): Input image or image path.
            auto (bool, optional): Whether to use rect.
            center_padding (bool, optional): Whether to center padding.
        Return:
            resized_img (numpy.ndarray): Image after resize and transpose,
                (H, W, C) -> (1, C, H, W).
        """
        if type(image) == str and os.path.isfile(image):
            img_raw = cv2.imread(image)
        else:
            img_raw = image
        resized_img, _, _ = letterbox(
            img_raw, new_shape=self.img_hw, auto=auto, center_padding=center_padding
        )
        if auto:
            self.img_hw = resized_img.shape[:2]
        # cv2.imshow('x', resized_img)
        # cv2.waitKey(0)

        # H, W, C -> 1, C, H, W
        resized_img = resized_img[:, :, ::-1].transpose(2, 0, 1)[None, :]
        resized_img = np.ascontiguousarray(resized_img)
        self.ori_hw.append(img_raw.shape[:2])
        return resized_img

    def preprocess_multi_img(self, images, auto=True, center_padding=True):
        """Preprocess multi image.

        Args:
            images (List[numpy.ndarray] | List[str]): Input images or image paths.
            auto (bool, optional): Whether to use rect.
            center_padding (bool, optional): Whether to center padding.
        Return:
            imgs (numpy.ndarray): Images after resize and transpose,
                List[(H, W, C)] -> (B, C, H, W).
        """
        resized_imgs = []
        for image in images:
            img = self.preprocess_one_img(
                image, auto=auto, center_padding=center_padding
            )
            resized_imgs.append(img)
        imgs = np.stack(resized_imgs, axis=0)
        return imgs

    def inference(self, images):
        """Inference.
        
        Args:
            images (numpy.ndarray | List[numpy.ndarray]): Input images.
        Return:
            see function `inference_single_model` and `inference_multi_model`.
        """
        if isinstance(images, list):
            imgs = self.preprocess_multi_img(images)
        else:
            imgs = self.preprocess_one_img(images)
        imgs = torch.from_numpy(imgs).to(self.device)
        imgs = imgs.half() if self.half else imgs.float()  # uint8 to fp16/32
        # if self.yolov5:
        #     imgs = imgs / 255.

        if self.multi_model:
            return self.inference_multi_model(imgs)
        else:
            return self.inference_single_model(imgs)

    def inference_single_model(self, images):
        """Inference single model.
        
        Args:
            images (torch.Tensor): B, C, H, W.
        Return:
            outputs (torch.Tensor): B, num_boxes, classes+5
        """
        if self.models.model_type == "yolov5":
            images = images / 255.0
        preds = self.models(images)
        if self.models.model_type == "yolov5":
            preds = preds[0]
        outputs = self.postprocess(
            preds,
            conf_thres=self.models.conf_thres,
            iou_thres=self.models.iou_thres,
            classes=self.models.filter,
        )
        return outputs

    def inference_multi_model(self, images):
        """Inference multi model.
        
        Args:
            images (torch.Tensor): B, C, H, W.
        Return:
            outputs (List[torch.Tensor]): List[B, num_boxes, classes+5]
        """
        total_outputs = []
        for mi, model in enumerate(self.models):
            inputs = images / 255.0 if self.yolov5[mi] else images
            preds = model(inputs)
            if model.model_type == "yolov5":
                preds = preds[0]
            total_outputs.append(
                self.postprocess(
                    preds,
                    conf_thres=model.conf_thres,
                    iou_thres=model.iou_thres,
                    classes=model.filter,
                )
            )
        return total_outputs

    def postprocess(self, preds, conf_thres=0.4, iou_thres=0.5, classes=None):
        """Postprocess multi images. NMS and scale coords to original image size.

        Args:
            preds (torch.Tensor): [B, num_boxes, classes+5].
        Return:
            otuputs (torch.Tensor): [B, num_boxes, classes+5].
        """
        outputs = non_max_suppression(
            preds, conf_thres, iou_thres, classes=classes, agnostic=False
        )
        for i, det in enumerate(outputs):  # detections per image
            if det is None or len(det) == 0:
                continue
            # TODO, suppert center_padding only.
            det[:, :4] = scale_coords(self.img_hw, det[:, :4], self.ori_hw[i]).round()
        return outputs
