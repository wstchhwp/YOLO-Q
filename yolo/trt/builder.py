from .yolov5 import YOLOV5TRT
from omegaconf import OmegaConf

def build_from_configs(cfg_path, cfx, stream):
    with open(cfg_path, 'r') as f:
        # config = yaml.safe_load(f)
        config =OmegaConf.load(f)

    model_list = []
    for _, v in config.items():
        model = YOLOV5TRT(engine_file_path=v.engine_file,
                          library=v.lib_file,
                          cfx=cfx,
                          stream=stream)
        setattr(model, 'conf_thres', v.conf_thres)
        setattr(model, 'iou_thres', v.iou_thres)
        setattr(model, 'filter', v.filter)
        setattr(model, 'names', v.names)
        model_list.append(model)
    if len(config) > 1:
        return model_list
    else:
        return model_list[0]
