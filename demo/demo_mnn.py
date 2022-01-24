""" python demo usage about MNN API """
import numpy as np
import MNN
import cv2
from yolo.data.data_reader import create_reader
from yolo.data.augmentations import letterbox

def inference(source='/e/1.avi'):
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("/d/projects/MNN/build/yolov5n.mnn")
    interpreter.setCacheFile('.tempcache')
    config = {}
    config['precision'] = 'low'
    
    # create session
    runtimeinfo, exists = MNN.Interpreter.createRuntime((config,))
    print(runtimeinfo, exists)
    session = interpreter.createSession(config, runtimeinfo)
    
    # show session info
    print('memory_info: %fMB' % interpreter.getSessionInfo(session, 0))
    print('flops_info: %fM' % interpreter.getSessionInfo(session, 1))
    print('backend_info: %d' % interpreter.getSessionInfo(session, 2))
    
    input_tensor = interpreter.getSessionInput(session)

    data = create_reader(source=source)
    #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((1, 15200, 85), MNN.Halide_Type_Float, np.ones([1, 15200, 85]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    for img, p, s in data:
        # print(s, p)
        ori_shape = img.shape[:2]
        resize_img, _, _ = letterbox(img, new_shape=(384, 640))

        resize_img = resize_img[:, :, ::-1].transpose(2, 0, 1)[None, :]
        resize_img = np.ascontiguousarray(resize_img)
        resize_img = resize_img.astype(np.float32)

        tmp_input = MNN.Tensor((1, 3, 384, 640), MNN.Halide_Type_Float,\
                        resize_img, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
        output_tensor = interpreter.getSessionOutput(session)
        output_tensor.copyToHostTensor(tmp_output)
        # print(len(tmp_output.getData()))
        # output = np.array(tmp_output.getData()).reshape(1, 15200, 85)

        cv2.imshow("p", img)
        if cv2.waitKey(1) == ord("q"):
            break
    # image = cv2.imread('ILSVRC2012_val_00049999.JPEG')
    # #cv2 read as bgr format
    # image = image[..., ::-1]
    # #change to rgb format
    # image = cv2.resize(image, (224, 224))
    # #resize to mobile_net tensor size
    # image = image - (103.94, 116.78, 123.68)
    # image = image * (0.017, 0.017, 0.017)
    # #preprocess it
    # image = image.transpose((2, 0, 1))
    # #change numpy data type as np.float32 to match tensor's format
    # image = image.astype(np.float32)
    # #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    # tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,\
    #                 image, MNN.Tensor_DimensionType_Caffe)
    # input_tensor.copyFrom(tmp_input)
    # interpreter.runSession(session)
    # output_tensor = interpreter.getSessionOutput(session)
    # #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    # tmp_output = MNN.Tensor((1, 1001), MNN.Halide_Type_Float, np.ones([1, 1001]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    # output_tensor.copyToHostTensor(tmp_output)
    # print("expect 983")
    # print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
if __name__ == "__main__":
    inference()
