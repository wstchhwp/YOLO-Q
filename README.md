# YOLO
A inference framework that support multi models of `yolo5`(torch and tensorrt), `yolox`(torch and tensorrt), `nanodet`(tensorrt), `yolo-fastestV2`(tensorrt).

## Requirement
- Tensorrt >= 8
- Torch >= 1.7.1

## Support
- [X] yolov5(torch, tensorrtx, tensorrt)
- [X] yolox(torch)
- [X] nanodet(tensorrt)

## TODO
- [ ] ~~preprocessing with right and bottom padding.~~
- [ ] ~~reference yolov5 class `AutoShape`, `Detections` and `Annotator`~~
- [X] same interface of `yolov5` and `yolox`.
- [X] **add time**
- [ ] add log
- [ ] ~~add `auto` to config~~
- [ ] better visualization
- [ ] clean up
- [ ] return the output with dict type
- [ ] device

## Quick Start
- Installation
  ```shell
  git clone https://github.com/Laughing-q/YOLO-Q.git
  pip install -r requirements.txt
  pip install -e .
  ```
- Export Model
  * yolov5
  ```shell
  git clone https://github.com/Laughing-q/yolov5-6.git
  python export.py --weights pt_file --include=engine --device 0 --imgsz h w [--half]
  ```
  * nanodet
  ```shell
  git clone https://github.com/Laughing-q/nanodet.git
  python tools/export_onnx.py --cfg_path cfg_file --model_path model_ckpt --out_path output_file --imgsz h w [--half]
  python tools/export_trt.py --onnx-path onnx_file [--half]
  ```
  * yolox
  ```shell
  git clone https://github.com/Laughing-q/YOLOX-Q.git
  python tools/export_onnx.py -f exp_file -c model_ckpt --imgsz h w [--half]
  python tools/export_trt.py --onnx-path onnx_file [--half]
  ```
  * yolo-fastestV2
  ```shell
  git clone https://github.com/Laughing-q/Yolo-FastestV2.git
  python pytorch2onnx.py --data data_file --weights model_ckpt --output output_file --imgsz h w [--half]
  python export_trt.py --onnx-path onnx_file [--half]
  ```

- Demo
  - prepare config file like below:
    * torch version
    ```vim
    model1:
      model_type: yolov5
      yaml: yolov5s.yaml
      weight: yolov5s.pth
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null

    model2:
      model_type: yolox
      yaml: yolox_nano.yaml
      weight: yolox_nano.pth
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
    ```
    * tensorrt version(tenxorrtx)
    ```vim
    model1:
      engine_file: yolov5n.engine
      lib_file: libmyplugins.so
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
      names: ./weights/trt/yolov5n.yaml

    model2:
      engine_file: yolov5n.engine
      lib_file: libmyplugins.so
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
      names: ./weights/trt/yolov5n.yaml
    ```
    * tensorrt version(onnx -> tensorrt)
    ```vim
    model1:
      model_type: yolov5
      engine_file: yolov5n.engine
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
      names: ./weights/trt/yolov5n.yaml

    model2:
      model_type: yolox
      engine_file: yolox-nano.engine
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
      names: ./weights/trt/yolov5n.yaml

    model3:
      model_type: nanodet
      engine_file: nanodet-plus.engine
      conf_thres: 0.4
      iou_thres: 0.4
      filter: null
      names: ./weights/trt/yolov5n.yaml
    ```
  - See `demo/` for more details.

## Tips
- The official weights of `yolo-fastestV2` and `nanodet-plus` trained with stretched(do not keep aspect ratio) images.
- The weights of `nanodet-plus` is compatible with stretching and kepping ratio images.
- The weights of `yolo-fastestV2` support stretching image only, or the prefermance will decline.

## Reference
- [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [https://github.com/open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)
