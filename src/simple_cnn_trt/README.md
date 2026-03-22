
# Simple CNN TensorRT Benchmark

## Goal

Validate TensorRT inference pipeline on Jetson Orin Nano.

Pipeline:


PyTorch → ONNX → TensorRT → Benchmark


---

## Model

Simple CNN

Input shape:


1 x 3 x 224 x 224


Precision:


FP16


---

## Export

```bash
python3 export_model.py

Output:

models/simple_cnn.onnx
TensorRT Engine
trtexec \
  --onnx=models/simple_cnn.onnx \
  --saveEngine=engines/simple_cnn_fp16.engine \
  --fp16
Benchmark
trtexec \
  --loadEngine=engines/simple_cnn_fp16.engine \
  --warmUp=1000 \
  --duration=10 \
  --iterations=50 \
  --useCudaGraph
Result
Metric	Value
Throughput	10677.2 qps
Mean Latency	0.1426 ms
P95 Latency	0.1445 ms
GPU Compute Time	0.0892 ms
Enqueue Time	0.0050 ms
Observations
High throughput (>10K QPS)
Low latency (~0.14 ms)
GPU compute dominates latency
Host overhead is minimal
Notes
FP16 precision was used for inference
TensorRT trtexec was used for engine generation and benchmarking
CUDA Graph was enabled to reduce launch overhead
