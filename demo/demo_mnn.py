""" python demo usage about MNN API """
import numpy as np
import MNN
import cv2
import torch
from lqcv.data.data_reader import create_reader
from yolo.data.augmentations import letterbox
from yolo.process.post import non_max_suppression
from yolo.api.visualization import Visualizer
from yolo.utils.boxes import scale_coords


def inference(source="/e/1.avi"):
    interpreter = MNN.Interpreter(
        "/d/projects/MNN/build/yolov5n.mnn"
    )
    # interpreter.setCacheFile(".tempcache")
    config = {}
    config["precision"] = "high"

    # create session
    # runtimeinfo, exists = MNN.Interpreter.createRuntime((config,))
    # print(runtimeinfo, exists)
    session = interpreter.createSession(config)

    # show session info
    print("memory_info: %fMB" % interpreter.getSessionInfo(session, 0))
    print("flops_info: %fM" % interpreter.getSessionInfo(session, 1))
    print("backend_info: %d" % interpreter.getSessionInfo(session, 2))

    input_tensor = interpreter.getSessionInput(session)
    vis = Visualizer(names=[i for i in range(80)])

    data = create_reader(source=source)
    # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    for img, p, s in data:
        # print(s, p)
        ori_shape = img.shape[:2]
        resize_img, _, _ = letterbox(img, new_shape=(384, 640), auto=False)

        resize_img = resize_img[:, :, ::-1].transpose(2, 0, 1)[None, :]
        resize_img = np.ascontiguousarray(resize_img)
        resize_img = resize_img.astype(np.float32)

        tmp_input = MNN.Tensor(
            (1, 3, 384, 640),
            MNN.Halide_Type_Float,
            resize_img,
            MNN.Tensor_DimensionType_Caffe,
        )
        input_tensor.copyFrom(tmp_input)
        # input_tensor.fromNumpy(resize_img)
        interpreter.runSession(session)
        output_tensor = interpreter.getSessionOutputAll(session)
        output_tensor = output_tensor['output']
        # tmp_output = MNN.Tensor(
        #     (1, 15120, 85),
        #     output_tensor.getDataType(),
        #     np.ones([1, 15120, 85]).astype(np.float32),
        #     output_tensor.getDimensionType(),
        # )
        # output_tensor.copyToHostTensor(tmp_output)
        # print(output_tensor.getShape())
        # print(output_tensor.getDataType())
        # print(output_tensor.getDimensionType())
        # print(len(output_tensor.getData()))
        pred = torch.Tensor(output_tensor.getData()).reshape(1, 15120, 85)
        outputs = non_max_suppression(pred, conf_thres=0.5)
        for i, det in enumerate(outputs):  # detections per image
            if det is None or len(det) == 0:
                continue
            det[:, :4] = scale_coords(
                [384, 640], det[:, :4], ori_shape, scale_fill=False
            ).round()
        print(outputs[0].shape)
        vis.draw_imgs(img, outputs)

        cv2.imshow("p", img)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    inference()
