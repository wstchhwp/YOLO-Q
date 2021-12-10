from detectors import Yolov5
import yolox
import cv2
import os.path as osp


if __name__ == "__main__":
    root = "./detectors/yolov5/weights"
    base = 'yolov5s'
    detector = Yolov5(cfg=osp.join(root, base + '.yaml'), 
                      weight_path=osp.join(root, base + '.pth'), 
                      device="0", 
                      img_hw=(640, 640))

    detector.show = True
    detector.pause = False

    # for video
    cap = cv2.VideoCapture('/e/1.avi')
    frame_num = 1
    while cap.isOpened():
        frame_num += 1
        # print(frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        img, img_raw = detector.preprocess(frame, auto=True)
        x = img_raw.copy()
        preds, _ = detector.dynamic_detect(img, [img_raw], conf_threshold=0.65)

    # for image
    # image = cv2.imread('1.png')
    # img, img_raw = detector.preprocess(image, auto=True)
    # preds, _ = detector.dynamic_detect(img, [img_raw])
