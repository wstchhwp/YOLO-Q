from easydict import EasyDict as edict

config = edict()

config.model1 = edict()
config.model1.model_type = 'yolov5'
config.model1.yaml = 'yolov5s.yaml'
config.model1.weight = 'yolov5s.pth'
config.model1.conf_thres = 0.4
config.model1.iou_thres = 0.4
config.model1.filter = [0, 1]

# config.model2 = edict()
# config.model2.model_type = 'yolox'
# config.model2.type = 's'
# config.model2.weight = 'yolox-nano.pth'
# config.model2.conf_thres = 0.4
# config.model2.iou_thres = 0.4
# config.model2.filter = [0, 1]
