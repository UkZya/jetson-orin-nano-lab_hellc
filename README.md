
# Jetson Orin Nano Lab

<div align="center">
  <img src="https://github.com/user-attachments/assets/a66a3d3d-9b9b-473e-b4c9-3fb43e4cf3d8" width="250" height="300" />
</div>

This repository documents the bring-up process of a **Jetson Orin Nano 8GB developer kit** and the setup of an edge AI development environment.

---

### 🌐 Networking Setup: Final Solution (LAN)
USB 네트워킹 대신 안정적인 LAN 연결로 전환한 최종 구성입니다.
`Jetson (LAN)` → `TP-Link Deco` → `Internet`

*   **IP 주소 확인:**
    ```bash
    ip a
    ```
*   **Static IP 설정 (Deco 앱):**
    *   **MAC Address:** `<your-MAC-address>`
    *   **Action:** TP-Link Deco 앱에서 해당 MAC 주소에 대해 **DHCP 예약(Reservation)** 설정.
    *   **Result:** 재부팅 후에도 항상 동일한 IP 할당 확인.

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
    sudo apt install nvidia-jetpack
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
2.  **Install PyTorch Wheel** (JetPack 호환 버전 필수)
    ```bash
    pip3 install torch-<jetson-wheel>.whl
    ```
3.  **CUDA Verification (Python)**
    ```python
    import torch
    x = torch.randn(1, 3, 224, 224, device='cuda')
    print(f"Shape: {x.shape}, Device: {x.device}")
    ```
    *   **Expected output:** `torch.Size([1, 3, 224, 224]) cuda:0`

---

### 📊 Monitoring Tools
*   **GPU/System:** `nvidia-smi`
*   **Jetson Info:** `sudo tegrastats`

---

### 🔍 Troubleshooting Notes

*   **`nvcc` not found**
    *   **Cause:** CUDA가 설치되었으나 환경 변수(`PATH`)가 설정되지 않음.
    *   **Fix:**
        ```bash
        export PATH=/usr/local/cuda/bin:$PATH
        ```
*   **PyTorch Import Errors**
    *   **Symptoms:** `libcudnn.so.8`, `libcudss.so.0` 등을 찾을 수 없음.
    *   **Solution:** 현재 JetPack 버전과 호환되지 않는 Wheel을 설치한 경우입니다. 공식 NVIDIA Jetson PyTorch Wheel을 사용하세요.

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
