- yolov5n
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

- yolov5s
    | yolov5s             | 640x384 | 640x384    |
    |---------------------|---------|------------|
    | Average preprocess  | 32.2ms  | **34.6ms** |
    | Average inference   | 78.4ms  | **71.5ms** |
    | Average postprocess | 26.1ms  | **26.6ms** |
    | Average memory      | 5107MB  | **4844MB** |
    | Average utilize     | 71%     | **66%**    |
    | Max utilize         | 82%     | **79%**    |
    | Tensorrt            | 7.1.3.4 | 8.2.1.8    |
