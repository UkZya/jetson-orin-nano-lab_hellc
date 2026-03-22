# 🚀 Simple CNN — TensorRT Inference Benchmark

> **Platform:** Jetson Orin Nano &nbsp;|&nbsp; **Precision:** FP16 &nbsp;|&nbsp; **Pipeline:** PyTorch → ONNX → TensorRT

---

## 📐 Model Specification

| Parameter | Value |
| :--- | :--- |
| Architecture | Simple CNN |
| Input Shape | `1 × 3 × 224 × 224` |
| Precision | FP16 |

---

## ⚙️ Pipeline

### 1. ONNX Export

```bash
python3 export_model.py
```

```
→ models/simple_cnn.onnx
```

### 2. TensorRT Engine Build

```bash
trtexec \
  --onnx=models/simple_cnn.onnx \
  --saveEngine=engines/simple_cnn_fp16.engine \
  --fp16
```

### 3. Benchmark

```bash
trtexec \
  --loadEngine=engines/simple_cnn_fp16.engine \
  --warmUp=1000 \
  --duration=10 \
  --iterations=50 \
  --useCudaGraph
```

---

## 📊 Results

| Metric | Value |
| :--- | ---: |
| **Throughput** | 10,677.2 qps |
| **Mean Latency** | 0.1426 ms |
| **P95 Latency** | 0.1445 ms |
| **GPU Compute Time** | 0.0892 ms |
| **Enqueue Time** | 0.0050 ms |

---

## 💡 Observations

- **High throughput** — 10K+ QPS, 충분한 실시간 처리 성능 확인
- **Low latency** — Mean ~0.14 ms, P95와의 편차 극히 미미 (jitter ≈ 0.002 ms)
- **GPU-bound** — GPU compute가 전체 latency의 ~63% 차지, host overhead 최소
- **CUDA Graph 효과** — Enqueue time 0.005 ms로 kernel launch overhead 대폭 절감

---

## 📝 Notes

- FP16 precision으로 inference 수행
- `trtexec`를 engine 생성 및 벤치마크에 사용
- `--useCudaGraph` 옵션으로 launch overhead 최소화
