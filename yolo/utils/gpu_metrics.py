import pynvml
import torch
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_mem_usage(pytorch=False):
    """
    Compute the GPU memory usage for the current device (MB).
    """
    if pytorch:
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    return mem_usage_bytes / (1024 * 1024)

def gpu_use():
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
