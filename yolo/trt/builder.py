from .yolov5 import YOLOV5TRT
from .trt_model import TRT_Model
from omegaconf import OmegaConf


def build_yolov5_trt(cfg, engine_file, library, ctx, stream):
    """build model from `tensorrtx" way"""
    with open(cfg, "r") as f:
        cfg = OmegaConf.load(f)
    model = YOLOV5TRT(
        engine_file_path=engine_file, library=library, ctx=ctx, stream=stream
    )
    setattr(model, "names", cfg.names)
    return model


def build_from_configs(config, ctx, stream):
    """build model from `tensorrtx" way"""
    if isinstance(config, str):
        with open(config, "r") as f:
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


def build_trt(cfg, engine_file, device):
    """build models from onnx way."""
    with open(cfg, "r") as f:
        cfg = OmegaConf.load(f)
    model = TRT_Model(engine_file_path=engine_file, device=device)  # TODO
    setattr(model, "names", cfg.names)
    return model


# TODO
def build_trt_from_configs(config):
    """build models from onnx way."""
    if isinstance(config, str):
        with open(config, "r") as f:
            config = OmegaConf.load(f)
    model_list = []
    device = config.get('device', '0')
    for _, v in config.items():
        model = build_trt(
            cfg=v.names,
            engine_file=v.engine_file,
            device=device
        )
        setattr(model, 'model_type', v.model_type)
        setattr(model, "conf_thres", v.conf_thres)
        setattr(model, "iou_thres", v.iou_thres)
        setattr(model, "filter", v.filter)
        model_list.append(model)
    if len(config) > 1:
        return model_list
    else:
        return model_list[0]
