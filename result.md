## Model Testing

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


- `yolov5n` vs `nanodet-plus-m_416` vs `yolox-nano`
  * num_camera x batch=5x15
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

  * num_camera x batch=5x1
    | Model               | yolov5n       | yolov5n       | yolox-nano    | yolox-tiny    | nanodet-plus-m_416 | nanodet-plus-m_416 | yolo-fastest  | yolov5-lite-s |
    |---------------------|---------------|---------------|---------------|---------------|--------------------|--------------------|---------------|---------------|
    | Input-size          | 5x1x3x640x384 | 5x1x3x416x256 | 5x1x3x640x384 | 5x1x3x640x384 | 5x1x3x640x384      | 5x1x3x416x416      | 5x1x3x640x384 | 5x1x3x640x384 |
    | Average preprocess  | 0.8ms         | 0.8ms         | 0.8ms         | 0.8ms         | 0.8ms              | 0.6ms              | 0.8ms         | 0.8ms         |
    | Average inference   | 3.9ms         | **3.1ms**     | 4.5ms         | 6.1ms         | 7.0(6.3)ms         | 6.3(5.5)ms         | 3.8ms         | 5.0ms         |
    | Average postprocess | 0.0ms         | 0.0ms         | 0.0ms         | 0.0ms         | 0.0ms              | 0.0ms              | 0.0ms         | 0.0ms         |
    | Average memory      | 1371MB        | **1351MB**    | 1359MB        | 1439MB        | 1387MB             | 1359MB             | 1366MB        | 1377MB        |
    | Average utilize     | 52.7%         | **47.7%**     | 58.4%         | 65.2%         | 65.2%              | 60.1%              | **49.9%**     | 61.9%         |
    | Max utilize         | 61%           | **56%**       | 65%           | 69%           | 68%                | 67%                | **55%**       | 65%           |
    | Tensorrt            | 8.2.1.8       | 8.2.1.8       | 8.2.1.8       | 8.2.1.8       | 8.2.1.8            | 8.2.1.8            | 8.2.1.8       | 8.2.1.8       |
  * num_camera x batch=5x1
    | Model               | yolov5-lite-g | yolov5-lite-c | yolov5s       |
    |---------------------|---------------|---------------|---------------|
    | Input-size          | 5x1x3x640x384 | 5x1x3x640x384 | 5x1x3x640x384 |
    | Average preprocess  | 0.8ms         | 0.8ms         | 0.8ms         |
    | Average inference   | 6.0ms         | 6.4ms         | **5.7ms**     |
    | Average postprocess | 0.0ms         | 0.0ms         | 0.0ms         |
    | Average memory      | 1473MB        | **1441MB**    | 1491MB        |
    | Average utilize     | 66%           | 66%           | **65.4%**     |
    | Max utilize         | 70%           | 71%           | **68%**       |
    | Tensorrt            | 8.2.1.8       | 8.2.1.8       | 8.2.1.8       |

- prune test
  * num_camera x batch=5x15
    | Model               | original model | prune model(-42%) |
    |---------------------|----------------|-------------------|
    | Input-size          | 5x15x3x640x384 | 5x15x3x640x384    |
    | Average preprocess  | 9.6ms          | 9.4ms             |
    | Average inference   | 22.8ms         | **22.5ms**        |
    | Average postprocess | 0.0ms          | 0.0ms             |
    | Average memory      | **1927MB**     | 1929MB            |
    | Average utilize     | 71.1%          | **70.9%**         |
    | Max utilize         | 72%            | 72%               |
    | Tensorrt            | 8.2.1.8        | 8.2.1.8           |
  * num_camera x batch=5x1
    | Model               | original model | prune model(-42%) |
    |---------------------|----------------|-------------------|
    | Input-size          | 5x1x3x640x384  | 5x1x3x640x384     |
    | Average preprocess  | 0.8ms          | 0.8ms             |
    | Average inference   | 3.4ms          | 3.4ms             |
    | Average postprocess | 0.0ms          | 0.0ms             |
    | Average memory      | 1371MB         | **1361MB**        |
    | Average utilize     | 48.6%          | 48.0%             |
    | Max utilize         | 57%            | 57%               |
    | Tensorrt            | 8.2.1.8        | 8.2.1.8           |
  * `-42%`表示剪枝后的模型减少了42%的参数量

- remove head test(yolov5n_p4)
  * num_camera x batch=5x15
    | Model               | original model | p4(-25.5%)     |
    |---------------------|----------------|----------------|
    | Input-size          | 5x15x3x640x384 | 5x15x3x640x384 |
    | Average preprocess  | 9.4ms          | 9.5ms          |
    | Average inference   | 20.0ms         | **18.3ms**     |
    | Average postprocess | 0.0ms          | 0.0ms          |
    | Average memory      | **2233MB**     | 2217MB         |
    | Average utilize     | 68.9%          | **66.5%**      |
    | Max utilize         | 69%            | **67%**        |
    | Tensorrt            | 8.2.1.8        | 8.2.1.8        |
  * num_camera x batch=5x1
    | Model               | original model | p4(-25.5%)    |
    |---------------------|----------------|---------------|
    | Input-size          | 5x1x3x640x384  | 5x1x3x640x384 |
    | Average preprocess  | 0.8ms          | 0.8ms         |
    | Average inference   | 3.3ms          | **2.7ms**     |
    | Average postprocess | 0.0ms          | 0.0ms         |
    | Average memory      | 1681MB         | **1679MB**    |
    | Average utilize     | 50.2%          | **44.3%**     |
    | Max utilize         | 58%            | **53%**       |
    | Tensorrt            | 8.2.1.8        | 8.2.1.8       |

- remove head test(yolov5n_p4_tiny)
  * num_camera x batch=5x15
    | Model               | original model | p4(-25.5%)     | p4_tiny(-73%)  |
    |---------------------|----------------|----------------|----------------|
    | Input-size          | 5x15x3x640x384 | 5x15x3x640x384 | 5x15x3x640x384 |
    | Average preprocess  | 9.5ms          | 9.5ms          | 9.5ms          |
    | Average inference   | 20.2ms         | 18.3ms         | **17.1ms**     |
    | Average postprocess | 0ms            | 0ms            | 0ms            |
    | Average memory      | 2601MB         | 2585MB         | **2563MB**     |
    | Average utilize     | 69%            | 66.9%          | **65.5%**      |
    | Max utilize         | 69%            | 68%            | **66%**        |
    | Tensorrt            | 8.2.1.8        | 8.2.1.8        | 8.2.1.8        |

  * num_camera x batch=5x1
    | Model               | original model | p4(-25.5%)    | p4_tiny(-73%) |
    |---------------------|----------------|---------------|---------------|
    | Input-size          | 5x1x3x640x384  | 5x1x3x640x384 | 5x1x3x640x384 |
    | Average preprocess  | 0.8ms          | 0.8ms         | 0.8ms         |
    | Average inference   | 3.3ms          | 2.7ms         | **2.2ms**     |
    | Average postprocess | 0ms            | 0ms           | 0ms           |
    | Average memory      | 2049MB         | 2047MB        | **2019MB**    |
    | Average utilize     | 47.6%          | 43.5%         | **38.6%**     |
    | Max utilize         | 57%            | 52%           | **48%**       |
    | Tensorrt            | 8.2.1.8        | 8.2.1.8       | 8.2.1.8       |

- 模型都是采用torch -> onnx -> engine的方式转tensorrt.
- `Input-size` = `num_camera` × `batch-size` × `w` × `h`.
- 去除了postprocess, 是为了排除其他的影响来测试gpu利用率，因为该代码库也采用gpu做postprocess.
- 测试流程是首先使用100frames作为warmup，然后计算500frames的平均值.
- 上表推理时间包含`normalize`操作(yolov5包含`images/255.`, yolox没有normalize).
- 由于nanodet的`normalize`操作是减均值除方差，所以会多花费一些时间，上表中nanodet后面的括弧数据为除去`normalize`的纯推理时间.
- 估计是由于讲anchor计算以及sigmoid写入了engine，所以类别越少速度越快.
- 测试设备: RTX2070super.

---
- 补充
  * 考虑到是为保持接口一致完善了nanodet中与anchor计算等操作，于是去掉该操作测试(保留了sigmoid等操作)
  * `nanodet-plus-m_416` + `5x1x3x640x384`
  * 7ms -> 6.4ms
  * 65.2% -> 63.9%
  * 总起来说区别不大
  * 为作对比, 将yolov5的相关计算也去掉
  * `yolov5` + `5x1x3x640x384`
  * 3.9ms -> 3.4ms
  * 52.7% -> 50.9%
  * 总起来说区别不大

## Reference
访存密集型vs计算密集型模型:[https://zhuanlan.zhihu.com/p/411522457](https://zhuanlan.zhihu.com/p/411522457)
