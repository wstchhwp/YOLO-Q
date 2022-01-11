#!/bin/sh

python demo/demo_trt_onnx.py --cfg-path ./configs/from_onnx/config_trt_onnx_n15.yaml &> "demo1n_onnx.log" &
python demo/demo_trt_onnx.py --cfg-path ./configs/from_onnx/config_trt_onnx_n15.yaml &> "demo2n_onnx.log" &
