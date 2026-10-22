
# Jetson Orin Nano Lab

<div align="center">
  <img src="https://github.com/user-attachments/assets/a66a3d3d-9b9b-473e-b4c9-3fb43e4cf3d8" width="250" height="300" />
</div>
This repository documents the bring-up process of a **Jetson Orin Nano 8GB Developer Kit** and the setup of an edge AI development environment.

---

### 🌐 Networking Setup: Final Solution (LAN)
Switched from unstable USB networking to a reliable LAN configuration.
`Jetson (LAN)` → `TP-Link Deco` → `Internet`

*   **Check IP Address:**
    ```bash
    ip a
    ```
*   **Static IP Configuration (via Deco App):**
    *   **MAC Address:** `<your-MAC-address>`
    *   **Action:** Set **DHCP Reservation** for this MAC address in the TP-Link Deco app.
    *   **Result:** Verified that the Jetson always receives the same IP after rebooting.

*   **SSH Workflow:**
    ```bash
    ssh namuk@<Your-Jetson-IP>
    ```

---

### 📦 JetPack / CUDA / TensorRT Setup

1.  **System Update**
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```
2.  **Install JetPack Components**
    ```bash
    sudo apt install nvidia-pack
    ```
3.  **Verify Installation**
    *   **CUDA:** `nvcc --version`
    *   **TensorRT:** `trtexec --help`

---

### 🔥 PyTorch Installation (Jetson)

1.  **Install Dependencies**
    ```bash
    sudo apt install libopenblas-base libopenmpi-dev libomp-dev
    ```
2.  **Install PyTorch Wheel** (Compatible JetPack version required)
    ```bash
    pip3 install torch-<jetson-wheel>.whl
    ```
3.  **CUDA Verification (Python)**
    ```python
    import torch
    x = torch.randn(1, 3, 224, 224, device='cuda')
    print(f"Shape: {x.shape}, Device: {x.device}")
    ```
    *   **Expected Output:** `torch.Size([1, 3, 224, 224]) cuda:0`

---

### 📊 Monitoring Tools
*   **GPU/System Management:** `nvidia-smi`
*   **Jetson Hardware Stats:** `sudo tegrastats`

---

### 🔍 Troubleshooting Notes

*   **`nvcc` not found**
    *   **Cause:** CUDA is installed, but the environment variable (`PATH`) is not configured.
    *   **Fix:**
        ```bash
        export PATH=/usr/local/cuda/bin:$PATH
        ```
*   **PyTorch Import Errors**
    *   **Symptoms:** Cannot find `libcudnn.so.8`, `libcudss.so.0`, etc.
    *   **Solution:** This happens when an incompatible Wheel for the current JetPack version is installed. Always use the official NVIDIA Jetson PyTorch Wheels.

---

### ⚡ YOLOv8n Benchmark: Power Mode & TensorRT

Inference speed of YOLOv8n (640px, batch 1) on a video file, comparing PyTorch (FP16) vs. TensorRT engine under two `nvpmodel` power modes.

| Model | Metric | 15W | MAXN_SUPER | Diff |
|---|---|---|---|---|
| `yolov8n.pt` (FP16) | End-to-end | 29.1 ms (34.4 FPS) | 25.7 ms (39.0 FPS) | MAXN_SUPER ~13% faster |
| `yolov8n.pt` (FP16) | Inference only | 21.7 ms | 19.0 ms | ~12% faster |
| `yolov8n.engine` (TensorRT) | End-to-end | 25.4 ms (39.4 FPS) | 25.4 ms (39.3 FPS) | no difference |
| `yolov8n.engine` (TensorRT) | Inference only | 11.9 ms | 11.8 ms | no difference |

**Takeaways**
*   TensorRT cuts pure inference time by ~1.6x (19.0 → 11.8 ms on MAXN_SUPER), but end-to-end time barely changes because CPU-side pre/post-processing (~14 ms) dominates.
*   PyTorch benefits from MAXN_SUPER (~12-13%), while the TensorRT engine runs the same in 15W and MAXN_SUPER. For a small model like YOLOv8n, 15W is enough, with roughly half the power and heat.

**Setup:** Jetson Orin Nano 8GB Super, JetPack 6.2.1 (L4T R36.4.7), CUDA 12.6, TensorRT 10.3, rootfs on NVMe SSD, `jetson_clocks` off.
Input: `videos/person-bicycle-car-detection.mp4` (768x432), 300 frames, 20 warm-up frames. Single run per mode (expect about ±1 ms noise).

**Reproduce**
```bash
sudo nvpmodel -m 0   # 15W  (-m 2 for MAXN_SUPER)
PYTHONPATH=~/trt_shim python src/bench_yolo.py yolov8n.pt yolov8n.engine \
    --source videos/person-bicycle-car-detection.mp4
```
Raw logs: [`runs/bench_15w.log`](runs/bench_15w.log), [`runs/bench_maxn_super.log`](runs/bench_maxn_super.log)

---

### 📂 Repository Structure
```text
jetson-orin-nano-lab
├── README.md
├── docs/
├── scripts/
│   ├── verify_cuda.sh
│   └── verify_torch_cuda.py
└── troubleshooting/
```

---

### 🎯 Next Steps

- [ ] PyTorch → ONNX export
- [ ] ONNX → TensorRT engine build
- [x] TensorRT latency benchmarking
- [ ] Edge perception pipeline experiments
