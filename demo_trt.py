from yolo.trt import build_from_configs
from yolo.api.trt_inference import TRTPredictor
from yolo.api.visualization import Visualizer
import cv2
import time
import pycuda.driver as cuda

if __name__ == "__main__":

    device = "0"
    cuda.init()
    cfx = cuda.Device(int(device)).make_context()
    stream = cuda.Stream()

    model = build_from_configs(cfg_path='./config_trt.yaml',
                               cfx=cfx,
                               stream=stream)
    predictor = TRTPredictor(img_hw=(384, 640), models=model, stream=stream)

    if predictor.multi_model:
        vis = [Visualizer(names=model.names) for model in predictor.models]
    else:
        vis = [Visualizer(names=predictor.models.names)]
        # vis.draw_imgs(img, outputs[i])
    # vis = Visualizer(names=model[1].names)

    cap = cv2.VideoCapture('/e/1.avi')
    frame_num = 1
    while cap.isOpened():
        st = time.time()
        frame_num += 1
        # if frame_num % 2 == 0:
        #     continue
        # print(frame_num)
        # if frame_num == 1000:
        #     break
        ret, frame = cap.read()
        frame_copy = frame.copy()
        if not ret:
            break
        outputs = predictor.inference([frame for _ in range(15)])
        # outputs = predictor.inference(frame)
        # for i, v in enumerate(vis):
        #     v.draw_imgs(frame, outputs[i])
        # cv2.imshow('p', frame)
        # # cv2.imshow('p1', frame.copy())
        # if cv2.waitKey(1) == ord('q'):
        #     break
        et = time.time()
        print(f'frame {frame_num} total time:', et - st)

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
