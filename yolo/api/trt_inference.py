from ..trt import to_device, alloc_inputs
from ..utils.boxes import nms_numpy, xywh2xyxy, scale_coords
from ..utils.timer import Timer
from ..process import normalize
from .inference import Predictor
import torch
from multiprocessing.pool import ThreadPool
import numpy as np
import torch.nn as nn
from typing import Optional


class TRTPredictorx(Predictor):  # `x` means repo `tensorrtx`
    """YOLOV5 Tensorrt Inference, support yolov5 model from repo `tensorrtx`"""

    def __init__(
        self,
        img_hw,
        models,
        stream,
        auto=False,
        pre_multi=False,
        infer_multi=False,
        post_multi=False,
    ):
        self.stream = stream
        self.img_hw = img_hw
        self.ori_hw = []
        self.models = models
        self.multi_model = True if isinstance(models, list) else False
        self.times = {}
        self.auto = auto

        # multi threading
        self.pre_multi = pre_multi
        self.infer_multi = infer_multi
        self.post_multi = post_multi
        # TODO
        self.sign = 1

    def preprocess(self, images):
        if isinstance(images, list):
            imgs = self.preprocess_multi_img(images)
        else:
            imgs = self.preprocess_one_img(images)

        if self.sign == 1:
            self.cuda_inputs, self.host_inputs = alloc_inputs(
                batch_size=len(imgs), hw=self.img_hw, split=False
            )
            self.sign += 1
        return imgs

    def inference_one_model(self, images, model=None):
        """Inference single model.

        Args:
            images (cuda): B, C, H, W.
        Return:
            outputs (List[numpy.ndarray]): List[num_boxes, 6] x B
        """
        if model is None:
            model = self.models
        preds = model(images)
        return preds

    def inference_multi_model(self, images):
        """Inference multi model.

        Args:
            images (torch.Tensor): B, C, H, W.
        Return:
            outputs (List[List[numpy.ndarray]]): List[List[num_boxes, 6] x B] x num_models
        """
        if self.infer_multi:
            # --------------multi threading-----------
            def single_infer(model):
                return self.inference_one_model(images, model)

            p = ThreadPool()
            total_outputs = p.map(single_infer, self.models)
            p.close()
        else:
            total_outputs = []
            for _, model in enumerate(self.models):
                preds = model(images)
                total_outputs.append(preds)
        return total_outputs

    def postprocess_one_model(self, preds, model=None):
        if model is None:
            model = self.models
        conf_thres = model.conf_thres
        iou_thres = model.iou_thres
        classes = model.filter
        out_preds = []
        for i, pred in enumerate(preds):
            if classes:
                pred = pred[(pred[:, 5:6] == classes).any(1)]
            si = pred[:, 4] > conf_thres
            pred = pred[si]
            pred[:, :4] = xywh2xyxy(pred[:, :4])
            # Do nms
            indices = nms_numpy(
                pred[:, :4], pred[:, 4], pred[:, 5], threshold=iou_thres
            )
            keep_pred = torch.from_numpy(pred[indices, :])
            if len(keep_pred):
                keep_pred[:, :4] = scale_coords(
                    self.img_hw, keep_pred[:, :4], self.ori_hw[i]
                ).round()
            else:
                keep_pred = None
            out_preds.append(keep_pred)
        if not self.multi_model:  # 表示只有一个模型
            self.ori_hw.clear()
        return out_preds

    def inference(self, images):
        """Inference.

        Args:
            images (numpy.ndarray | List[numpy.ndarray]): Input images.
        Return:
            see function `inference_single_model` and `inference_multi_model`.
        """
        preprocess = self.preprocess
        forward = (
            self.inference_multi_model if self.multi_model else self.inference_one_model
        )
        postprocess = (
            self.postprocess_multi_model
            if self.multi_model
            else self.postprocess_one_model
        )

        timer = Timer(cuda_sync=False)
        imgs = preprocess(images)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = to_device(
            imgs, self.host_inputs, self.cuda_inputs, self.stream, split=False
        )
        self.times["preprocess"] = round(timer.since_start() * 1000, 1)

        preds = forward(imgs)
        self.times["inference"] = round(timer.since_last_check() * 1000, 1)
        outputs = postprocess(preds)
        self.times["postprocess"] = round(timer.since_last_check() * 1000, 1)
        self.times["total"] = round(timer.since_start() * 1000, 1)

        return outputs


class TRTPredictor(Predictor):
    """Support yolov5 model from onnx -> tensorrt"""

    def __init__(
        self,
        img_hw,
        models,
        device,
        half=True,
        auto=False,
        pre_multi=False,
        infer_multi=False,
        post_multi=False,
    ):
        super(TRTPredictor, self).__init__(
            img_hw, models, device, half, auto, pre_multi, infer_multi, post_multi
        )

    def inference_one_model(
        self, images: torch.Tensor, model: Optional[nn.Module] = None
    ):
        """Inference single model.

        Args:
            images (torch.Tensor): B, C, H, W.
        Return:
            outputs (List[torch.Tensor]): List[num_boxes, classes+5] x B
        """
        model = self.models if model is None else model
        images = normalize[model.model_type](images)
        preds = model(images)
        return preds
