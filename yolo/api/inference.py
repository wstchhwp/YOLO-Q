from ..data.datasets import letterbox
from ..utils.boxes import scale_coords
from ..process import nms
from ..utils.general import to_2tuple
from ..utils.torch_utils import select_device
from ..utils.timer import Timer, time_sync
from ..process import normalize
from multiprocessing.pool import ThreadPool
from typing import Optional
import os
import cv2
import numpy as np
import torch.nn as nn
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

    def __init__(
        self,
        img_hw,
        models,
        device,
        half=True,
        auto=True,
        pre_multi=False,
        infer_multi=False,
        post_multi=False,
    ):
        super(Predictor, self).__init__()
        img_hw = to_2tuple(img_hw) if isinstance(img_hw, int) else img_hw

        self.img_hw = img_hw
        self.ori_hw = []
        self.models = models
        self.device = select_device(device)
        self.half = half
        self.auto = auto
        self.multi_model = True if isinstance(models, list) else False

        # multi threading
        self.pre_multi = pre_multi
        self.infer_multi = infer_multi
        self.post_multi = post_multi
        self._create_thread()

        # times
        self.timer = Timer(start=False, cuda_sync=True, round=1, unit="ms")
        self.times = {}

    def _create_thread(self):
        self.p = (
            ThreadPool()
            if self.pre_multi or self.infer_multi or self.post_multi
            else None
        )

    def preprocess_one_img(self, image, center_padding=True):
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
            img_raw,
            new_shape=self.img_hw,
            auto=self.auto,
            scaleFill=False,
            center_padding=center_padding,
        )
        if self.auto:
            self.img_hw = resized_img.shape[:2]
        # cv2.imshow('x', resized_img)
        # cv2.waitKey(0)

        # H, W, C -> 1, C, H, W
        resized_img = resized_img[:, :, ::-1].transpose(2, 0, 1)[None, :]
        resized_img = np.ascontiguousarray(resized_img)
        self.ori_hw.append(img_raw.shape[:2])
        return resized_img

    def preprocess_multi_img(self, images, center_padding=True):
        """Preprocess multi image.

        Args:
            images (List[numpy.ndarray] | List[str]): Input images or image paths.
            auto (bool, optional): Whether to use rect.
            center_padding (bool, optional): Whether to center padding.
        Return:
            imgs (numpy.ndarray): Images after resize and transpose,
                List[(H, W, C)] -> (B, C, H, W).
        """

        if self.pre_multi:
            # --------------multi threading-----------
            def single_pre(image):
                return self.preprocess_one_img(image, center_padding=center_padding)

            # p = ThreadPool()
            resized_imgs = self.p.map(single_pre, images)
            # p.close()
        else:
            # --------------single threading-----------
            resized_imgs = []
            for image in images:
                img = self.preprocess_one_img(image, center_padding=center_padding)
                resized_imgs.append(img)
        imgs = np.concatenate(resized_imgs, axis=0)
        return imgs

    def inference_one_model(
        self, images: torch.Tensor, model: Optional[nn.Module] = None
    ):
        """Inference single model.

        Args:
            images (torch.Tensor): B, C, H, W.
        Return:
            outputs (List[torch.Tensor]): List[num_boxes, classes+5] x B
        """
        # TODO
        model = self.models if model is None else model
        # normalize on gpu may be faster, cause copy int8 from cpu to gpu is faster.
        # `normalize` here is for support different models in one config file.
        # this may add a bit of time cost on `inference time`.
        images = normalize[model.model_type](images)
        images = images.half() if self.half else images.float()  # uint8 to fp16/32

        preds = model(images)
        if model.model_type == "yolov5":
            preds = preds[0]
        return preds

    def inference_multi_model(self, images: torch.Tensor) -> list:
        """Inference multi model.

        Args:
            images (torch.Tensor): B, C, H, W.
        Return:
            outputs (List[List[torch.Tensor]]): List[List[num_boxes, classes+5] x B] x num_models
        """

        if self.infer_multi:
            # --------------multi threading-----------
            def single_infer(model):
                return self.inference_one_model(images, model)

            # p = ThreadPool()
            total_outputs = self.p.map(single_infer, self.models)
            # p.close()
        else:
            # --------------single threading-----------
            total_outputs = []
            for _, model in enumerate(self.models):
                preds = self.inference_one_model(images, model)
                total_outputs.append(preds)
        return total_outputs

    def postprocess_one_model(self, preds, model: Optional[nn.Module] = None):
        """Postprocess multi images. NMS and scale coords to original image size.

        Args:
            preds (torch.Tensor): [B, num_boxes, classes+5].
        Return:
            otuputs (List[torch.Tensor]): List[torch.Tensor(num_boxes, 6)]xB.
        """
        if model is None:
            model = self.models
        conf_thres = model.conf_thres
        iou_thres = model.iou_thres
        classes = model.filter
        outputs = nms[model.model_type](
            preds, conf_thres, iou_thres, classes=classes, agnostic=False
        )
        for i, det in enumerate(outputs):  # detections per image
            if det is None or len(det) == 0:
                continue
            # TODO, suppert center_padding only.
            det[:, :4] = scale_coords(
                self.img_hw, det[:, :4], self.ori_hw[i], scale_fill=False
            ).round()
        # TODO
        if not self.multi_model:  # 表示只有一个模型
            self.ori_hw.clear()
        return outputs

    def postprocess_multi_model(self, outputs: list):
        """Postprocess multi images. NMS and scale coords to original image size.

        Args:
            outputs (List[torch.Tensor]): List[B, num_boxes, classes+5].
            models (List[nn.Module]): Models.
        Return:
            outputs (List[List[torch.Tensor]]): List[List[torch.Tensor(num_boxes, 6)]*B]*num_models,
                results after nms and scale_coords.
        """

        if self.post_multi:
            # --------------multi threading-----------
            def single_post(i):
                output = outputs[i]
                model = self.models[i]
                outputs[i] = self.postprocess_one_model(output, model)

            # p = ThreadPool()
            self.p.map(single_post, range(len(self.models)))
            # p.close()
        else:
            # --------------single threading-----------
            for i in range(len(outputs)):
                outputs[i] = self.postprocess_one_model(outputs[i], self.models[i])
        # TODO
        self.ori_hw.clear()
        return outputs

    def inference(self, images, post=True):
        """Inference.

        Args:
            images (numpy.ndarray | List[numpy.ndarray]): Input images.
            post (Bool): Whether to do postprocess, may be useful for some test situation.
        Return:
            see function `inference_single_model` and `inference_multi_model`.
        """
        preprocess = (
            self.preprocess_multi_img
            if isinstance(images, list)
            else self.preprocess_one_img
        )
        forward = (
            self.inference_multi_model if self.multi_model else self.inference_one_model
        )
        postprocess = (
            self.postprocess_multi_model
            if self.multi_model
            else self.postprocess_one_model
        )

        self.timer.start(reset=True)
        imgs = preprocess(images)
        imgs = torch.from_numpy(imgs).to(self.device)
        # cause some `normalize` operation maybe out of index,
        # so put it to func `inference_one_model.`
        # imgs = imgs.half() if self.half else imgs.float()  # uint8 to fp16/32
        self.times["preprocess"] = self.timer.since_last_check()

        preds = forward(imgs)
        self.times["inference"] = self.timer.since_last_check()
        if post:
            outputs = postprocess(preds)
        self.times["postprocess"] = self.timer.since_last_check()
        self.times["total"] = self.timer.since_start()

        return outputs if post else preds
