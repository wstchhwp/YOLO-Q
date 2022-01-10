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

- `yolov5n` vs `nanodet-plus-m_416`
  | Model               | yolov5n        | nanodet-plus-m_416 | yolov5n       | nanodet-plus-m_416 | nanodet-plus-m_416 |
  |---------------------|----------------|--------------------|---------------|--------------------|--------------------|
  | Input-size          | 5x15x3x640x384 | 5x15x3x640x384     | 5x1x3x640x384 | 5x1x3x640x384      | 5x1x3x416x416      |
  | Average preprocess  | 14.3ms         | 12.5ms             | 2.8ms         | 3.7ms              | 2.6ms              |
  | Average inference   | **32.3ms**     | 56.1ms             | **4.3ms**     | 7.8ms              | 6.9ms              |
  | Average postprocess | 0.0ms          | 0.0ms              | 0.0ms         | 0.0ms              | 0.0ms              |
  | Average memory      | **2933MB**     | 2643MB             | 1939MB        | 1965MB             | **1927MB**         |
  | Average utilize     | **72%**        | 80.8%              | **52.7%**     | 67.0%              | 66.1%              |
  | Max utilize         | 73%            | 82%                | 53%           | 68%                | 67%                |
  | Tensorrt            | 8.2.1.8        | 8.2.1.8            | 8.2.1.8       | 8.2.1.8            | 8.2.1.8            |
  * 这两个模型都是采用torch -> onnx -> engine的方式转tensorrt.
  * `Input-size` = `num_camera` × `batch-size` × `w` × `h`.
  * 去除了postprocess, 是为了排除其他的影响来测试gpu利用率，因为该代码库也采用gpu做postprocess.
  * 测试流程是首先使用100frames作为warmup，然后计算500frames的平均值.
