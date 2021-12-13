from yolo.trt import YOLOV5TRT, to_device, alloc_inputs
from yolo.data.augmentations import letterbox
from yolo.utils.boxes import nms_numpy, xywh2xyxy
from yolo.utils.plots import plot_one_box
import cv2
import numpy as np
import pycuda.driver as cuda

img = cv2.imread('/home/laughing/Screenshots/pic-full-211213-1444-39.png')

img, _, _ = letterbox(img, (640, 640), auto=False)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image = image.astype(np.float32)
# Normalize to [0,1]
image /= 255.0
# HWC to CHW format:
image = np.transpose(image, [2, 0, 1])
# CHW to NCHW format
image = np.expand_dims(image, axis=0)
# Convert the image to row-major order, also known as "C order":
image = np.ascontiguousarray(image)

cuda.init()
cfx = cuda.Device(0).make_context()
stream = cuda.Stream()
cuda_inputs, host_inputs = alloc_inputs(batch_size=1,
                                        hw=(640, 640),
                                        split=False)

image = to_device(image, host_inputs, cuda_inputs, stream, split=False)

model = YOLOV5TRT(
    engine_file_path='/home/laughing/yolov5/tensorrtx/buildn640/yolov5n.engine',
    library='/home/laughing/yolov5/tensorrtx/buildn640/libmyplugins.so', 
    cfx=cfx,
    stream=stream)

output = model(image)[0]
scores = output[:, 4]
output = output[scores > 0.2]

output[:, :4] = xywh2xyxy(output[:, :4])

boxes = output[:, :4]
cls = output[:, 5]
scores = output[:, 4]

index = nms_numpy(boxes, scores, cls, 0.5)
output = output[index]

for out in output:
    box = out[:4]
    cls = out[5]
    plot_one_box(box, img)

cfx.pop()

cv2.imshow('p', img)
cv2.waitKey(0)
