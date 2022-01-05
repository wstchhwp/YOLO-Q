from .yolov5 import YOLOV5TRT
from .trt_model import TRT_Model
from omegaconf import OmegaConf


def build_yolov5_trt(cfg, engine_file, library, ctx, stream):
    with open(cfg, "r") as f:
        cfg = OmegaConf.load(f)
    model = YOLOV5TRT(
        engine_file_path=engine_file, library=library, ctx=ctx, stream=stream
    )
    setattr(model, "names", cfg.names)
    return model


def build_trt(cfg, engine_file):
    with open(cfg, "r") as f:
        cfg = OmegaConf.load(f)
    model = TRT_Model(engine_file_path=engine_file, device="0")  # TODO
    setattr(model, "names", cfg.names)
    return model


# TODO
def build_trt_from_configs(cfg_path):
    with open(cfg_path, "r") as f:
        config = OmegaConf.load(f)
    model_list = []
    for _, v in config.items():
        model = build_trt(
            cfg=v.names,
            engine_file=v.engine_file,
        )
        setattr(model, "conf_thres", v.conf_thres)
        setattr(model, "iou_thres", v.iou_thres)
        setattr(model, "filter", v.filter)
        model_list.append(model)
    if len(config) > 1:
        return model_list
    else:
        return model_list[0]


def build_from_configs(cfg_path, ctx, stream):
    with open(cfg_path, "r") as f:
        # config = yaml.safe_load(f)
        config = OmegaConf.load(f)

    model_list = []
    for _, v in config.items():
        model = build_yolov5_trt(
            cfg=v.names,
            engine_file=v.engine_file,
            library=v.lib_file,
            ctx=ctx,
            stream=stream,
        )
        setattr(model, "conf_thres", v.conf_thres)
        setattr(model, "iou_thres", v.iou_thres)
        setattr(model, "filter", v.filter)
        model_list.append(model)
    if len(config) > 1:
        return model_list
    else:
        return model_list[0]
