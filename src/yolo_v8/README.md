# YOLOv8n Profiling on Jetson Orin Nano

Benchmarking and profiling results for **YOLOv8n** across multiple inference backends on Jetson Orin Nano.

---

## 📁 Project Structure

```
├── yolo_v8/
    ├── 00_onnx_export.py
    ├── 1_single_whole.py
    ├── 2_multiple_whole.py
    ├── 3_3stage_separate.py
    ├── 4_forward_only.py
    ├── 5_module_separate.py
    ├── 6_profiler.py
    ├── 7_onnx_profile.py
    ├── 8_tensorrt_fp16.py
    └── 9_tensorrt_inference.py
```

---

## 🚀 Backend Comparison Summary

| Backend | Avg Latency | Min Latency | Max Latency | FPS |
|---|---|---|---|---|
| **PyTorch (Full Pipeline)** | 40.91 ms | 39.30 ms | 48.23 ms | 24.45 |
| **PyTorch (Forward Only)** | 28.00 ms | 27.65 ms | 35.75 ms | **35.71** |
| **ONNX Runtime** | 385.64 ms | 191.41 ms | 725.77 ms | 2.59 |
| **TensorRT FP16** | **6.12 ms** | — | — | **183.5 qps** |

> ✅ TensorRT FP16 is **~6.7× faster** than PyTorch forward-only, and **~63× faster** than ONNX Runtime.

---

## 📊 PyTorch Stage-wise Breakdown

> Full pipeline: preprocess → inference → postprocess

| Stage | Avg | Min | Max | Ratio |
|---|---|---|---|---|
| Preprocess | 5.61 ms | 5.16 ms | 6.31 ms | 17.50% |
| **Inference** | **22.41 ms** | 21.67 ms | 22.80 ms | **69.94%** |
| Postprocess | 4.02 ms | 3.93 ms | 4.17 ms | 12.56% |
| **Total** | **32.03 ms** | 30.81 ms | 32.95 ms | 100% |

---

## 🔬 ONNX Runtime Breakdown

| Stage | Avg | Min | Max |
|---|---|---|---|
| Preprocess | 12.89 ms | 7.16 ms | 25.89 ms |
| **Inference** | **372.75 ms** | 181.60 ms | 708.62 ms |
| **Total** | **385.64 ms** | 191.41 ms | 725.77 ms |

> ⚠️ ONNX Runtime shows high variance (181 ms ~ 708 ms). CPU fallback ops suspected.

---

## ⚡ TensorRT FP16 Result

| Metric | Value |
|---|---|
| Mean Latency | 6.12 ms |
| GPU Compute Time | 5.41 ms |
| Throughput | 183.5 qps |

**Build command:**
```bash
trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=engines/yolov8n_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --verbose
```

**Benchmark command:**
```bash
trtexec \
  --loadEngine=engines/yolov8n_fp16.engine \
  --warmUp=100 \
  --iterations=200 \
  --useCudaGraph
```

---

## 🧩 Module-wise Profiling (Top 20, sorted by total time)

| Module | Avg (ms) | Total (ms) | Calls |
|---|---|---|---|
| model.0.act | 0.165 | 281.558 | 1710 |
| model.0.conv | 6.007 | 180.212 | 30 |
| model.2.cv1.conv | 3.090 | 92.708 | 30 |
| model.1.conv | 2.538 | 76.140 | 30 |
| model.22.cv3.0.1.conv | 1.671 | 50.124 | 30 |
| model.2.m.0.cv1.conv | 1.630 | 48.907 | 30 |
| model.22.cv3.0.0.conv | 1.345 | 40.341 | 30 |
| model.4.m.0.cv1.conv | 1.169 | 35.063 | 30 |
| model.0.bn | 1.160 | 34.810 | 30 |
| model.22.cv2.0.0.conv | 1.141 | 34.238 | 30 |
| model.22.dfl.conv | 1.110 | 33.289 | 30 |
| model.7.conv | 1.076 | 32.269 | 30 |
| model.3.conv | 1.048 | 31.453 | 30 |
| model.4.m.1.cv1.conv | 1.007 | 30.201 | 30 |
| model.4.m.1.cv2.conv | 1.006 | 30.168 | 30 |
| model.4.m.0.cv2.conv | 1.003 | 30.102 | 30 |
| model.22.cv2.0.1.conv | 1.001 | 30.035 | 30 |
| model.15.m.0.cv1.conv | 0.989 | 29.684 | 30 |
| model.15.m.0.cv2.conv | 0.973 | 29.191 | 30 |
| model.22.cv2.1.0.conv | 0.906 | 27.167 | 30 |

---

## 🔥 Op-wise Profiling — Top 15 (CPU time domain)

| Op | Time (ms) | Ratio | Calls |
|---|---|---|---|
| aten::conv2d | 164.916 | 14.19% | 1280 |
| aten::convolution | 159.206 | 13.70% | 1280 |
| aten::_convolution | 149.525 | 12.86% | 1280 |
| aten::batch_norm | 136.808 | 11.77% | 1140 |
| aten::_batch_norm_impl_index | 130.964 | 11.27% | 1140 |
| aten::cudnn_convolution | 129.083 | 11.11% | 1280 |
| aten::cudnn_batch_norm | 122.864 | 10.57% | 1140 |
| aten::silu_ | 40.121 | 3.45% | 1140 |
| aten::empty | 32.992 | 2.84% | 4580 |
| aten::empty_like | 19.803 | 1.70% | 1160 |
| aten::cat | 15.859 | 1.36% | 340 |
| aten::chunk | 7.336 | 0.63% | 180 |
| aten::add | 6.674 | 0.57% | 160 |
| aten::split | 6.491 | 0.56% | 180 |
| aten::view | 5.973 | 0.51% | 1420 |

---

## 🎯 Optimization Targets (Priority Order)

### 1순위 — Conv + BN Fusion
| Op | Time (ms) | Ratio |
|---|---|---|
| `aten::cudnn_convolution` | 129.083 ms | 11.11% |
| `aten::cudnn_batch_norm` | 122.864 ms | 10.57% |

→ Conv-BN fusion (e.g., `torch.fx` or TensorRT graph optimization) 으로 직접 절감 가능

### 2순위 — Activation
| Op | Time (ms) | Ratio |
|---|---|---|
| `aten::silu_` | 40.121 ms | 3.45% |

→ SiLU → ReLU 교체 or fused kernel 적용 검토

### 3순위 — Memory Allocation
| Op | Time (ms) | Ratio |
|---|---|---|
| `aten::empty` | 32.992 ms | 2.84% |
| `aten::empty_like` | 19.803 ms | 1.70% |
| `aten::cat` | 15.859 ms | 1.36% |

→ 잦은 메모리 할당 → pre-allocated buffer / in-place op 활용 권장

---

## 🖥️ Environment

| Item | Detail |
|---|---|
| Device | Jetson Orin Nano |
| Model | YOLOv8n |
| TensorRT Precision | FP16 |
| CUDA Graph | Enabled (trtexec) |
