- yolov5n
  * 640x384比640x640少很多资源占用，速度更快
  * 以下为两个进程跑15x5的结果(也就是30x5)
    | yolov5n             | 640x640 | 640x384 |
    |---------------------|---------|---------|
    | Average preprocess  | 48.9ms  | 34.5ms  |
    | Average inference   | 59.4ms  | 38.2ms  |
    | Average postprocess | 23.1ms  | 23.9ms  |
    | Average memory      | 5047MB  | 4213MB  |
    | Average utilize     | 69%     | 62%     |
    | Max utilize         | 84%     | 72%     |

- yolov5s
  | yolov5s             | 640x384 |
  |---------------------|---------|
  | Average preprocess  | 32.2ms  |
  | Average inference   | 78.4ms  |
  | Average postprocess | 26.1ms  |
  | Average memory      | 5107MB  |
  | Average utilize     | 71%     |
  | Max utilize         | 82%     |
