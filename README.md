
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
    *   **MAC Address:** `<your Jetson's MAC address>` (check with `ip link`)
    *   **Action:** Set **DHCP Reservation** for this MAC address in the TP-Link Deco app.
    *   **Result:** Verified that the Jetson always receives the same IP after rebooting.

*   **SSH Workflow:**
    ```bash
    ssh <user>@<Your-Jetson-IP>
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

### 💾 Moving the Root Filesystem from SD to NVMe SSD

The Orin Nano firmware keeps booting from the **SD card's** `/boot/extlinux/extlinux.conf`; only the root filesystem (`root=`) moves to the SSD. The SD card stays as a fallback.

1.  **Format and mount the SSD** (ext4, e.g. label `ssd`, mounted at `/mnt/ssd`).
2.  **Copy the rootfs** (about 32 GB used):
    ```bash
    sudo rsync -axHAWXS --numeric-ids --info=progress2 --exclude='/mnt/*' / /mnt/ssd/
    ```
    `-x` stays on one filesystem, so the SSD itself and `/proc`, `/sys` are not copied.
3.  **Find the SSD partition's PARTUUID** (more reliable than `UUID=` for the kernel's `root=`):
    ```bash
    lsblk -o NAME,UUID,PARTUUID /dev/nvme0n1
    ```
4.  **Edit the SD card's `extlinux.conf`**: back it up, add an entry with `root=PARTUUID=<PARTUUID>`, make it `DEFAULT`, and keep the old entry (`root=/dev/mmcblk0p1`) as `SD fallback`.
5.  **Reboot and verify**:
    ```bash
    findmnt /        # /dev/nvme0n1p1
    ```
6.  **Clean up** the now redundant `/mnt/ssd` line in `/etc/fstab`.

**Gotchas**
*   After booting from the SSD, `/boot/extlinux/extlinux.conf` is a **stale copy** from the time of the rsync. Editing it (or running `jetson-io.py`) has no effect, because the firmware reads the SD card's file. To change boot settings, mount the SD partition (`sudo mount /dev/mmcblk0p1 /mnt/sd`) and edit `/mnt/sd/boot/extlinux/extlinux.conf`. Kernel updates may only touch the stale copy, so re-check the SD side after updating.
*   The conda environment and PyTorch CUDA worked unchanged after the move. The `nvpmodel` setting (MAXN_SUPER) also persisted across reboots from the SSD.
*   Keep the SD fallback entry and its card for a few days until the SSD boot proves stable.

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

### ⚡ YOLOv8n Benchmark

YOLOv8n latency across backends and power modes (15W vs MAXN_SUPER, PyTorch vs TensorRT) is documented in [`src/yolo_v8/README.md`](src/yolo_v8/README.md).

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
