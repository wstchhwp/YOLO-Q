from yolo.trt import build_from_configs
from yolo.api.trt_inference import TRTPredictor
from yolo.api.visualization import Visualizer
from yolo.utils.metrics import MeterBuffer
from yolo.utils.gpu_metrics import gpu_mem_usage, gpu_use
import cv2
import pycuda.driver as cuda
from loguru import logger
import time

if __name__ == "__main__":

    device = "0"
    cuda.init()
    cfx = cuda.Device(int(device)).make_context()
    stream = cuda.Stream()

    pre_multi = False  # 多线程速度较快
    infer_multi = False  # 路数较少时，多线程速度较快
    post_multi = False  # 多线程速度较快

    # logger.add("trt15.log", format="{message}")
    # logger.add("trt1.log", format="{message}")
    # logger.add("trt15.log")

    model = build_from_configs(cfg_path='./configs/config_trt.yaml',
                               cfx=cfx,
                               stream=stream)
    predictor = TRTPredictor(
        img_hw=(384, 640),
        models=model,
        stream=stream,
        pre_multi=pre_multi,
        infer_multi=infer_multi,
        post_multi=post_multi,
    )

    if predictor.multi_model:
        vis = [Visualizer(names=model.names) for model in predictor.models]
    else:
        vis = [Visualizer(names=predictor.models.names)]
        # vis.draw_imgs(img, outputs[i])
    # vis = Visualizer(names=model[1].names)

    meter = MeterBuffer(window_size=100)

    cap = cv2.VideoCapture('/e/1.avi')
    frame_num = 1
    while cap.isOpened():
        ts = time.time()
        frame_num += 1
        # if frame_num % 2 == 0:
        #     continue
        if frame_num == 200:
            break
        ret, frame = cap.read()
        if not ret:
            break
        # outputs = predictor.inference([frame for _ in range(15)])
        outputs = predictor.inference(frame)
        for i, v in enumerate(vis):
            v.draw_imgs(frame, outputs[i])
        cv2.imshow('p', frame)
        # cv2.imshow('p1', frame.copy())
        if cv2.waitKey(1) == ord('q'):
            break
        te = time.time()
        print(f"frame {frame_num} time: {te - ts}")
        memory = gpu_mem_usage()
        utilize = gpu_use()
        logger.info(f"{predictor.times}, {memory}, {utilize}")
        meter.update(memory=memory, utilize=utilize, **predictor.times)

    logger.info("-------------------------------------------------------")
    # logger.info("Tensort, 15x5, yolov5n, 640x384, 100/200frames average time.")
    logger.info("Tensort, 1x5, yolov5n, 640x384, 100/200frames average time.")
    logger.info(f"pre_multi: {pre_multi}")
    logger.info(f"infer_multi: {infer_multi}")
    logger.info(f"post_multi: {post_multi}")
    logger.info(f"Average preprocess: {meter['preprocess'].avg}s")
    logger.info(f"Average inference: {meter['inference'].avg}s")
    logger.info(f"Average postprocess: {meter['postprocess'].avg}s")
    logger.info(f"Average memory: {meter['memory'].avg}MB")
    logger.info(f"Average utilize: {meter['utilize'].avg}%")

cfx.pop()

# -----------5000 frames-----------
# -----------two models------------
# single thread total time: 85.99742603302002
# multi thread total time: 66.30611062049866

# -----------three models------------
# single thread total time: 117.53678607940674
# multi thread total time: 78.62954616546631

# -----------three models, two pic------------
# single thread total time: 136.21081161499023
# multi thread total time: 107.52954616546631

# -----------1000 frames-----------
# -----------four model, two pic-----------
# single thread total time: 43.65745544433594
# multi thread total time: 32.65745544433594
