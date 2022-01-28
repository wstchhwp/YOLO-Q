from yolo.trt import build_trt_from_configs
from yolo.api.trt_inference import TRTPredictor
from yolo.api.visualization import Visualizer
from yolo.utils.metrics import MeterBuffer
from yolo.utils.general import parse_config
from yolo.utils.gpu_metrics import gpu_mem_usage, gpu_use
import cv2
from loguru import logger
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Demo of yolov5(tensorrt from onnx).",
    )
    parser.add_argument(
        "--cfg-path",
        default="./configs/from_onnx/config_trt_onnx_n.yaml",
        type=str,
        help="Path to .yml config file.",
    )
    parser.add_argument(
        "--post", action='store_true', help="Whether do nms."
    )
    parser.add_argument(
        "--show", action='store_true', help="Whether to visualize and show frame."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cfg_path = args.cfg_path
    model_config, settings = parse_config(cfg_path)

    pre_multi = settings.pre_multi  # batch-size较大时设置为True会有提速
    infer_multi = settings.infer_multi  # 多线程速度较慢
    post_multi = settings.post_multi  # 多线程速度较慢

    show = args.show
    post = args.post
    if show:
        assert post, "You should set `post`=True."

    warmup_frames = 100
    test_frames = 500
    # setting = global_settings[cfg_path]

    test_batch = settings.batch_size
    test_model = settings.test_title
    test_size = settings.input_hw

    # logger.add("trt15.log", format="{message}")
    # logger.add("trt1.log", format="{message}")
    # logger.add("trt15.log")

    model = build_trt_from_configs(config=model_config)
    predictor = TRTPredictor(
        img_hw=test_size,
        models=model,
        device=0,
        auto=True,
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

    meter = MeterBuffer(window_size=500)

    cap = cv2.VideoCapture("/e/1.avi")
    frames = warmup_frames + test_frames
    pbar = tqdm(range(frames), total=frames)
    for frame_num in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        outputs = predictor.inference([frame for _ in range(test_batch)], post=post)
        if show:
            for i, v in enumerate(vis):
                v.draw_imgs(frame, outputs[i], vis_confs=0.2)
            cv2.imshow("p", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        memory = gpu_mem_usage()
        utilize = gpu_use()
        pbar.desc = f"{predictor.times}, {memory}, {utilize}"
        # logger.info(f"{predictor.times}, {memory}, {utilize}")
        meter.update(memory=memory, utilize=utilize, **predictor.times)
    predictor.close_thread()
    logger.info("-------------------------------------------------------")
    logger.info(
        f"Tensort, {test_batch}x5, {test_model}, {test_size}, {test_frames}frames average time."
    )
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
