- `yolov5n`
  * 640x384比640x640少很多资源占用，速度更快
  * 以下为两个进程跑15x5的结果(也就是30x5)
    | yolov5n             | 640x640 | 640x384 | 640x640 | 640x384    |
    |---------------------|---------|---------|---------|------------|
    | Average preprocess  | 48.9ms  | 34.5ms  | 45.7ms  | 37.0ms     |
    | Average inference   | 59.4ms  | 38.2ms  | 46.7ms  | **29.0ms** |
    | Average postprocess | 23.1ms  | 23.9ms  | 23.4ms  | 23.7ms     |
    | Average memory      | 5047MB  | 4213MB  | 4916MB  | **4170MB** |
    | Average utilize     | 69%     | 62%     | 73%     | **58%**    |
    | Max utilize         | 84%     | 72%     | 78%     | **65%**    |
    | Tensorrt            | 7.1.3.4 | 7.1.3.4 | 7.1.3.4 | 8.2.1.8    |

- `yolov5s`
    | yolov5s             | 640x384 | 640x384    |
    |---------------------|---------|------------|
    | Average preprocess  | 32.2ms  | **34.6ms** |
    | Average inference   | 78.4ms  | **71.5ms** |
    | Average postprocess | 26.1ms  | **26.6ms** |
    | Average memory      | 5107MB  | **4844MB** |
    | Average utilize     | 71%     | **66%**    |
    | Max utilize         | 82%     | **79%**    |
    | Tensorrt            | 7.1.3.4 | 8.2.1.8    |

TODO
- `yolov5n` vs `nanodet-plus-m_416` vs `yolox-nano`
  * batch=5x15
    | Model               | yolov5n        | nanodet-plus-m_416 | yolox-nano     | yolov5n        |
    |---------------------|----------------|--------------------|----------------|----------------|
    | Input-size          | 5x15x3x640x384 | 5x15x3x640x384     | 5x15x3x640x384 | 5x15x3x416x256 |
    | Average preprocess  | 12.5ms         | 12.3ms             | 12.3ms         | 6.8ms          |
    | Average inference   | 31.7ms         | 54.3(51.2)ms       | 33.0ms         | **14.9ms**     |
    | Average postprocess | 0.0ms          | 0.0ms              | 0.0ms          | 0.0ms          |
    | Average memory      | 2365MB         | 2075MB             | 1839MB         | **1669MB**     |
    | Average utilize     | 72%            | 79.1%              | 71.6%          | **64.6%**      |
    | Max utilize         | 73%            | 80%                | 73%            | **65%**        |
    | Tensorrt            | 8.2.1.8        | 8.2.1.8            | 8.2.1.8        | 8.2.1.8        |

  * batch=5x1
    | Model               | yolov5n       | yolov5n       | yolox-nano    | yolox-tiny    | nanodet-plus-m_416 | nanodet-plus-m_416 | yolo-fastest  |
    |---------------------|---------------|---------------|---------------|---------------|--------------------|--------------------|---------------|
    | Input-size          | 5x1x3x640x384 | 5x1x3x416x256 | 5x1x3x640x384 | 5x1x3x640x384 | 5x1x3x640x384      | 5x1x3x416x416      | 5x1x3x640x384 |
    | Average preprocess  | 0.8ms         | 0.8ms         | 0.8ms         | 0.8ms         | 0.8ms              | 0.6ms              | 0.8ms         |
    | Average inference   | 3.9ms         | **3.1ms**     | 4.5ms         | 6.1ms         | 7.0(6.3)ms         | 6.3(5.5)ms         | 3.8ms         |
    | Average postprocess | 0.0ms         | 0.0ms         | 0.0ms         | 0.0ms         | 0.0ms              | 0.0ms              | 0.0ms         |
    | Average memory      | 1371MB        | **1351MB**    | 1359MB        | 1439MB        | 1387MB             | 1359MB             | 1366MB        |
    | Average utilize     | 52.7%         | **47.7%**     | 58.4%         | 65.2%         | 65.2%              | 60.1%              | **49.9%*      |
    | Max utilize         | 61%           | 56%           | 65%           | 69%           | 68%                | 67%                | **55%**       |
    | Tensorrt            | 8.2.1.8       | 8.2.1.8       | 8.2.1.8       | 8.2.1.8       | 8.2.1.8            | 8.2.1.8            | 8.2.1.8       |
  * 这两个模型都是采用torch -> onnx -> engine的方式转tensorrt.
  * `Input-size` = `num_camera` × `batch-size` × `w` × `h`.
  * 去除了postprocess, 是为了排除其他的影响来测试gpu利用率，因为该代码库也采用gpu做postprocess.
  * 测试流程是首先使用100frames作为warmup，然后计算500frames的平均值.
  * 上表推理时间包含`normalize`操作(yolov5包含`images/255.`, yolox没有normalize).
  * 由于nanodet的`normalize`操作是减均值除方差，所以会多花费一些时间，上表中nanodet后面的括弧数据为除去`normalize`的纯推理时间.
  * 测试设备: RTX2070super.
