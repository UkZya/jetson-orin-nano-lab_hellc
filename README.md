



# Jetson Orin Nano Lab

<div align="center">
  <img src="https://github.com/user-attachments/assets/a66a3d3d-9b9b-473e-b4c9-3fb43e4cf3d8" width="250" height="300" />
</div>

This repository documents the bring-up process of a **Jetson Orin Nano 8GB developer kit** and the setup of an edge AI development environment using **NVIDIA JetPack, CUDA, TensorRT, and PyTorch**.

The goal of this repository is to create a reproducible development environment for **edge AI experiments** such as TensorRT optimization and perception pipelines.

---

## 🛠 Hardware
* **Device:** Jetson Orin Nano 8GB Developer Kit
* **Storage:** microSD boot
* **Connectivity:** LAN connection (TP-Link Deco network)
* **Host PC:** Windows 11

---

## 💻 Software Stack
* **OS:** Jetson Linux (JetPack 6.2) / Ubuntu 22.04
* **Libraries:** 
    * CUDA 12.6
    * TensorRT 10.x
    * PyTorch (Jetson compatible wheel)
* **Language:** Python 3.10

---

## 🚀 Initial Setup

### 1. Flash Jetson Linux
1. Download the Jetson Orin Nano SD card image from NVIDIA.
2. Flash the image using **balenaEtcher**.
3. Insert the microSD card into the Jetson slot.
4. Power on the device.

### 2. Headless Setup (Serial Console)
Since no monitor was available, the initial configuration was performed via **serial console**.

* **Connection:** `PC USB-A` → `Jetson USB-C`
* **Tool:** PuTTY (or Serial terminal)
* **Settings:** 
    * Serial COM Port
    * Speed: **115200 baud**

Complete the Ubuntu setup through the serial terminal.

---

## 🌐 Networking Setup

### Attempt 1: USB Networking (Windows ICS)
Initial attempt used USB networking between Windows and Jetson.
* **Issues encountered:**
    * DHCP configuration failure
    * Gateway not assigned
    * Windows ICS routing issues
    * USB gadget networking conflicts

> [!CAUTION]
> **Conclusion:** USB networking was unreliable for Jetson development.

### Final Solution: LAN Networking
Switched to a stable LAN connection.
`Jetson (LAN)` → `TP-Link Deco` → `Internet`

* **Check IP address:**
  ```bash
  ip a  
Static IP via Deco:

Jetson MAC address: 4c:bb:47:62:14:60

Configured DHCP reservation in TP-Link Deco.

Result: Jetson always receives the same IP address.

SSH Workflow
After LAN setup, development is performed through SSH, removing the need for serial or USB connections.

Bash
ssh namuk@<IP Address>
📦 JetPack / CUDA / TensorRT Setup
1. Update System
Bash
sudo apt update && sudo apt upgrade -y
2. Install JetPack Components
Bash
sudo apt install nvidia-jetpack
3. Verify Installation
CUDA: nvcc --version

TensorRT: trtexec --help

🔥 PyTorch Installation (Jetson)
Jetson requires a JetPack-compatible PyTorch wheel.

1. Install Dependencies
Bash
sudo apt install libopenblas-base libopenmpi-dev libomp-dev
2. Install PyTorch Wheel
Bash
pip3 install torch-<jetson-wheel>.whl
3. CUDA Verification (Python)
Python
import torch
x = torch.randn(1, 3, 224, 224, device='cuda')
print(f"Shape: {x.shape}, Device: {x.device}")
Expected output: torch.Size([1, 3, 224, 224]) cuda:0

📊 Monitoring Tools
NVIDIA System Management: nvidia-smi

Jetson Specific Stats: sudo tegrastats

🔍 Troubleshooting Notes
nvcc not found
Cause: CUDA installed but PATH not configured.

Fix:

Bash
export PATH=/usr/local/cuda/bin:$PATH
PyTorch Import Errors
Symptoms: libcudnn.so.8, libcudss.so.0, or libcusparseLt.so.0 not found.

Cause: Incompatible PyTorch wheels for the current JetPack version.

Solution: Always use official NVIDIA JetPack-compatible PyTorch wheels.

📂 Repository Structure
Plaintext
jetson-orin-nano-lab
├── README.md
├── docs/
├── scripts/
│   ├── verify_cuda.sh
│   └── verify_torch_cuda.py
└── troubleshooting/
🎯 Next Steps
[ ] PyTorch → ONNX export

[ ] ONNX → TensorRT engine build

[ ] TensorRT latency benchmarking

[ ] Edge perception pipeline experiments
