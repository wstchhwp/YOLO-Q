import pynvml
import torch
import os
pynvml.nvmlInit()
# TODO
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_mem_usage(pytorch=False):
    """
    Compute the GPU memory usage for the current device (MB).
    """
    if pytorch:
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    return round(mem_usage_bytes / (1024 * 1024))

def gpu_use():
    return round(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)

def get_total_and_free_memory_in_Mb(cuda_device: int):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)

def get_python_memory_in_Mb(cuda_device: int):
    devices_info_str = os.popen(
        f"nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,nounits,noheader -i {cuda_device}"
    )
    total = 0
    devices_info = devices_info_str.read()
    if len(devices_info) == 0:
        return total
    devices_info = devices_info.strip().split("\n")
    for p in devices_info:
        name, used = p.split(",")
        if name == 'python':
            total += int(used)
    return total

def get_gpu_utilization(cuda_device: int):
    devices_info_str = os.popen(
        f"nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader -i {cuda_device}"
    )
    devices_info = devices_info_str.read()
    if len(devices_info) == 0:
        return 0
    return int(devices_info)


if __name__ == "__main__":
    print(get_python_memory_in_Mb(0))
    print(get_gpu_utilization(0))
