#!/bin/sh

python demo/demo_trt_onnx.py &> "demo1n_onnx.log" &
python demo/demo_trt_onnx.py &> "demo2n_onnx.log" &
