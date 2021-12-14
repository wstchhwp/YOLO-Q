from ..trt import to_device, alloc_inputs
from ..utils.boxes import nms_numpy, xywh2xyxy, scale_coords
from ..utils.timer import Timer
from .inference import Predictor
import torch
from multiprocessing.pool import ThreadPool
import numpy as np


class TRTPredictor(Predictor):
    """YOLOV5 Tensorrt Inference"""
    def __init__(self,
                 img_hw,
                 models,
                 stream,
                 pre_multi=False,
                 infer_multi=False,
                 post_multi=False):
        self.stream = stream
        self.img_hw = img_hw
        self.ori_hw = []
        self.models = models
        self.multi_model = True if isinstance(models, list) else False
        self.times = {}

        # multi threading
        self.pre_multi = pre_multi
        self.infer_multi = infer_multi
        self.post_multi = post_multi
        # TODO
        self.sign = 1

    def preprocess(self, images, auto=True):
        if isinstance(images, list):
            imgs = self.preprocess_multi_img(images, auto)
        else:
            imgs = self.preprocess_one_img(images, auto)

        if self.sign == 1:
            self.cuda_inputs, self.host_inputs = alloc_inputs(
                batch_size=len(imgs), hw=self.img_hw, split=False)
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
            indices = nms_numpy(pred[:, :4],
                                pred[:, 4],
                                pred[:, 5],
                                threshold=iou_thres)
            keep_pred = torch.from_numpy(pred[indices, :])
            if len(keep_pred):
                keep_pred[:, :4] = scale_coords(self.img_hw, keep_pred[:, :4],
                                                self.ori_hw[i]).round()
            else:
                keep_pred = None
            out_preds.append(keep_pred)
        return out_preds

    def inference(self, images):
        """Inference.

        Args:
            images (numpy.ndarray | List[numpy.ndarray]): Input images.
        Return:
            see function `inference_single_model` and `inference_multi_model`.
        """
        preprocess = self.preprocess
        forward = (self.inference_multi_model
                   if self.multi_model else self.inference_one_model)
        postprocess = (self.postprocess_multi_model
                       if self.multi_model else self.postprocess_one_model)

        timer = Timer(cuda_sync=False)
        imgs = preprocess(images)
        imgs = imgs.astype(np.float32) / 255.
        imgs = to_device(imgs,
                         self.host_inputs,
                         self.cuda_inputs,
                         self.stream,
                         split=False)
        self.times['preprocess'] = timer.since_start()

        preds = forward(imgs)
        self.times['inference'] = timer.since_last_check()
        outputs = postprocess(preds)
        self.times['postprocess'] = timer.since_last_check()

        return outputs
