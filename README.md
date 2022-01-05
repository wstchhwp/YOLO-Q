# yolo
A inference framework that support multi models of `yolo5`(torch and tensorrt) and `yolox`(torch).

- [ ] preprocessing with right and bottom padding.
- [ ] reference yolov5 class `AutoShape`, `Detections` and `Annotator`
- [X] same interface of `yolov5` and `yolox`.
- [X] **add time**
- [ ] add log
- [ ] add `auto` to config
- [ ] better visualization
- [ ] clean up
- [ ] inference
- [ ] return the output with dict type
- [ ] device

## Quick Start
- Installation
  ```shell
  pip install -e .
  ```
- Demo
  - prepare config file like below:
    * torch version
    ```vim
    model1:
      model_type: yolov5
      yaml: ./weights/yolov5/yolov5s.yaml
      weight: ./weights/yolov5/yolov5s.pth
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null

    model2:
      model_type: yolox
      yaml: ./weights/yolox/yolox_nano.yaml
      weight: ./weights/yolox/yolox_nano.pth
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
    ```
    * tensorrt version
    ```vim
    model1:
      engine_file: /home/laughing/yolov5/tensorrtx/buildn/yolov5n.engine
      lib_file: /home/laughing/yolov5/tensorrtx/buildn/libmyplugins.so
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
      names: ./weights/trt/yolov5n.yaml

    model2:
      engine_file: /home/laughing/yolov5/tensorrtx/buildn/yolov5n.engine
      lib_file: /home/laughing/yolov5/tensorrtx/buildn/libmyplugins.so
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
      names: ./weights/trt/yolov5n.yaml
    ```
  - then run `demo.py` or `demo_trt.py`, more details see `demo.py` or `demo_trt.py`.

## Reference
- [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [https://github.com/open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)
