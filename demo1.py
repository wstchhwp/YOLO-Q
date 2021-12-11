from yolo.models import build_yolov5
from yolo.api.inference import Predictor
from yolo.api.visualization import Visualizer
import os.path as osp
import cv2

if __name__ == "__main__":
    root = "./weights"
    base = 'yolov5s'
    model = build_yolov5(cfg=osp.join(root, base + '.yaml'), 
                         weight_path=osp.join(root, base + '.pth'),
                         device="0")
    predictor = Predictor(img_hw=(640, 640),
                          models=model,
                          device="0",
                          model_type='yolov5',
                          half=True)
    # if predictor.multi_model:
    #     vis = [Visualizer(names=model.classnames) for model in predictor.models]
    #     for i, v in enumerate(vis):
    #         v.draw_imgs(img, outputs[i])
    # else:
    #     vis = Visualizer(names=predictor.models.classnames)
    #     vis.draw_imgs(img, outputs[i])
    vis = Visualizer(names=model.classnames)

    cap = cv2.VideoCapture('/e/1.avi')
    frame_num = 1
    while cap.isOpened():
        frame_num += 1
        # print(frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        outputs = predictor.inference(frame)
        img = vis.draw_one_img(frame, outputs)
        cv2.imshow('p', img)
        if cv2.waitKey(1) == ord('q'):
            break
