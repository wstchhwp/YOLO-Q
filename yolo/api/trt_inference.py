from ..trt import to_device, alloc_inputs
from ..data.augmentations import letterbox
from ..utils.boxes import nms_numpy, xywh2xyxy, scale_coords
import cv2
import numpy as np
import torch
import os
import time


class TRTPredictor(object):
    """YOLOV5 Tensorrt Inference"""
    def __init__(self, img_hw, models, stream):
        self.stream = stream
        self.img_hw = img_hw
        self.ori_hw = []
        self.models = models
        self.multi_model = True if isinstance(models, list) else False
        # TODO
        self.sign = 1

    def preprocess(self, images, auto=True):
        if isinstance(images, list):
            imgs = self.preprocess_multi_img(images, auto)
        else:
            imgs = self.preprocess_one_img(images, auto)

        if self.sign == 1:
            self.cuda_inputs, self.host_inputs = alloc_inputs(batch_size=len(imgs),
                                                              hw=self.img_hw,
                                                              split=False)
            self.sign += 1
        return imgs

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
        resized_img, _, _ = letterbox(img_raw,
                                      new_shape=self.img_hw,
                                      auto=auto,
                                      center_padding=center_padding)
        if auto:
            self.img_hw = resized_img.shape[:2]
        # cv2.imshow('x', resized_img)
        # cv2.waitKey(0)

        # H, W, C -> 1, C, H, W
        resized_img = resized_img[:, :, ::-1].transpose(2, 0, 1)[None, :]
        resized_img = np.ascontiguousarray(resized_img)
        # TODO
        resized_img = resized_img.astype(np.float32) / 255.0
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
            img = self.preprocess_one_img(image,
                                          auto=auto,
                                          center_padding=center_padding)
            resized_imgs.append(img)
        imgs = np.concatenate(resized_imgs, axis=0)
        return imgs

    def inference(self, images):
        pst = time.time()
        imgs = self.preprocess(images)
        pet = time.time()
        images = to_device(imgs,
                           self.host_inputs,
                           self.cuda_inputs,
                           self.stream,
                           split=False)
        print('preprocess time:', pet - pst)
        if self.multi_model:
            return self.inference_multi_model(images)
            # return self.inference_multi_thread(images)
        else:
            return self.inference_single_model(images)

    def inference_single_model(self, images):
        """Inference single model.
        
        Args:
            images (cuda): B, C, H, W.
        Return:
            outputs (List[numpy.ndarray]): List[num_boxes, 6] x B
        """
        preds = self.models(images)
        # postprocessing
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
            outputs (List[List[numpy.ndarray]]): List[List[num_boxes, 6] x B] x num_models
        """
        total_outputs = []
        for _, model in enumerate(self.models):
            mt = time.time()
            preds = model(images)
            me = time.time()
            print('inference time:', me - mt)
            total_outputs.append(preds)
            # total_outputs.append(
            #     self.postprocess(
            #         preds,
            #         conf_thres=model.conf_thres,
            #         iou_thres=model.iou_thres,
            #         classes=model.filter,
            #     ))
        return total_outputs

    def postprocess(self,
                       preds,
                       conf_thres=0.4,
                       iou_thres=0.5,
                       classes=None):
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
        # return pred[indices, :]
        return out_preds
