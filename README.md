
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
    *   **MAC Address:** `4c:bb:47:62:14:60`
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
- [ ] TensorRT latency benchmarking
- [ ] Edge perception pipeline experiments
