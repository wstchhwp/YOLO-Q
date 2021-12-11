from yolo.models import build_yolov5
from yolo.api.inference import Predictor
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

    cap = cv2.VideoCapture('/e/1.avi')
    frame_num = 1
    while cap.isOpened():
        frame_num += 1
        # print(frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        outputs = predictor.inference(frame)
        img = predictor.visualize_one_img(frame, outputs)
        cv2.imshow('p', img)
        if cv2.waitKey(1) == ord('q'):
            break
