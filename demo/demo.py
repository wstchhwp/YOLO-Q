from yolo.models import build_from_configs
from yolo.api.inference import Predictor
from yolo.api.visualization import Visualizer
from yolo.utils.metrics import MeterBuffer
from yolo.utils.gpu_metrics import gpu_mem_usage, gpu_use
from loguru import logger
import cv2
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Demo of yolov5(torch).",
    )
    parser.add_argument(
        "--cfg-path",
        default="./configs/config.yaml",
        type=str,
        help="Path to .yml config file.",
    )
    parser.add_argument(
        "--show", action='store_true', help="Model intput shape."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # logger.add("torch15.log", format="{message}")
    # logger.add("torch1.log", format="{message}")
    args = parse_args()

    pre_multi = True    # 多线程速度较快
    infer_multi = True  # 路数较少时，多线程速度较快
    post_multi = True   # 多线程速度较快

    show = args.show

    model = build_from_configs(cfg_path=args.cfg_path)
    predictor = Predictor(
        img_hw=(640, 640),
        models=model,
        device="0",
        half=True,
        pre_multi=pre_multi,
        infer_multi=infer_multi, 
        post_multi=post_multi,
    )
    if predictor.multi_model:
        vis = [Visualizer(names=model.names) for model in predictor.models]
    else:
        vis = [Visualizer(names=predictor.models.names)]

    meter = MeterBuffer(window_size=100)

    cap = cv2.VideoCapture('/e/1.avi')
    frame_num = 1
    while cap.isOpened():
        st = time.time()
        frame_num += 1
        # if frame_num % 2 == 0:
        #     continue
        # print(frame_num)
        if frame_num == 500:
            break
        ret, frame = cap.read()
        frame_copy = frame.copy()
        if not ret:
            break
        outputs = predictor.inference([frame for _ in range(1)])
        if show:
            for i, v in enumerate(vis):
                v.draw_imgs(frame, outputs[i])
            cv2.imshow('p', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        memory = gpu_mem_usage()
        utilize = gpu_use()
        logger.info(f"{predictor.times}, {memory}, {utilize}")
        meter.update(memory=memory, utilize=utilize, **predictor.times)

    logger.info("-------------------------------------------------------")
    # logger.info("Torch, 15x5, yolov5n, 640x384, 200frames average time.")
    logger.info("Torch, 1x5, yolov5n, 640x384, 100/500frames average time.")
    logger.info(f"pre_multi: {pre_multi}")
    logger.info(f"infer_multi: {infer_multi}")
    logger.info(f"post_multi: {post_multi}")
    logger.info(f"Average preprocess: {round(meter['preprocess'].avg, 1)}ms")
    logger.info(f"Average inference: {round(meter['inference'].avg, 1)}ms")
    logger.info(f"Average postprocess: {round(meter['postprocess'].avg, 1)}ms")
    logger.info(f"Average Total: {round(meter['total'].avg, 1)}ms")
    logger.info(f"Average memory: {round(meter['memory'].avg)}MB")
    logger.info(f"Average utilize: {round(meter['utilize'].avg, 1)}%")
    logger.info(f"Max utilize: {round(meter['utilize'].max, 1)}%")

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
