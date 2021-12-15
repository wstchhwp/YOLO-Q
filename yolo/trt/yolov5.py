import numpy as np
import ctypes
import pycuda.driver as cuda
import tensorrt as trt

def alloc_inputs(batch_size, hw, split=True):
    """
    split:表示以batch-size为1处理
    """
    host_inputs = []
    cuda_inputs = []
    dtype = np.float32
    if split:
        for _ in range(batch_size):
            size = hw[0] * hw[1] * 3
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # []是为了和其他情况调用一致，detect里面是使用的cuda_input[0]. host_input[0]
            host_inputs.append([host_mem])
            cuda_inputs.append([cuda_mem])
    else:
        size = hw[0] * hw[1] * 3 * batch_size
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        # []是为了和其他情况调用一致，detect里面是使用的cuda_input[0]. host_input[0]
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    return cuda_inputs, host_inputs


def to_device(input_image, host_inputs, cuda_inputs, stream, split=True):
    """
    split:表示以batch-size为1处理
    """
    # input_image = input_image / 255.
    if split:
        for i, img in enumerate(input_image):
            np.copyto(host_inputs[i][0], img.ravel())
            cuda.memcpy_htod_async(
                cuda_inputs[i][0], host_inputs[i][0], stream)
    else:
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(
            cuda_inputs[0], host_inputs[0], stream)
    return cuda_inputs


class YOLOV5TRT:
    def __init__(self, engine_file_path, library, ctx, stream):
        # Create a Context on this device,
        self.ctx = ctx
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        ctypes.CDLL(library)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        # print(engine)
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        # self.bindings = bindings

        self.batch_size = engine.max_batch_size
        self.alloc_output(batch_size=self.batch_size)

    def alloc_input(self, batch_size=0, hw=(384, 640)):
        batch_size = self.engine.max_batch_size if batch_size == 0 else batch_size
        size = hw[0] * hw[1] * 3 * batch_size
        dtype = np.float32
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        self.host_inputs.append(host_mem)
        self.cuda_inputs.append(cuda_mem)

    def alloc_output(self, batch_size=0, size=6001):
        batch_size = self.engine.max_batch_size if batch_size == 0 else batch_size
        size = size * batch_size
        dtype = np.float32
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        self.host_outputs.append(host_mem)
        self.cuda_outputs.append(cuda_mem)

    def __call__(self, cuda_input):
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        # host_inputs = self.host_inputs
        cuda_inputs = cuda_input
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        # Run inference.
        context.execute_async(batch_size=self.batch_size,
                              bindings=[cuda_inputs[0], cuda_outputs[0]], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        preds = np.split(output, self.batch_size)  # list
        preds = [np.reshape(pred[1:], (-1, 6))[:int(pred[0]), :]
                 for pred in preds]
        return preds

    def destory(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
