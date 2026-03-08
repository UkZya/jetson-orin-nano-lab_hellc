# Jetson Orin Nano Lab

<div align="center">
  <img src="https://github.com/user-attachments/assets/a66a3d3d-9b9b-473e-b4c9-3fb43e4cf3d8" width="250" height="300" />
</div>
This repository documents the bring-up process of a Jetson Orin Nano 8GB developer kit and the setup of an edge AI development environment using NVIDIA JetPack, CUDA, TensorRT, and PyTorch.

The goal of this repository is to build a reproducible Jetson environment for future edge AI experiments such as TensorRT optimization and perception pipelines.

---


## Hardware

- Jetson Orin Nano 8GB Developer Kit
- microSD boot
- LAN connection
- TP-Link Deco network

---

## Software

- Jetson Linux (JetPack 6.2)
- CUDA 12.6
- TensorRT 10.x
- PyTorch 2.10.0 (Jetson wheel)
- Python 3.10

---

## What was done

1. Flash Jetson Linux image to microSD
2. Perform initial headless setup through serial console
3. Configure USB networking and Windows ICS
4. Switch to LAN-based networking
5. Configure DHCP reservation using TP-Link Deco
6. Install JetPack components (CUDA / TensorRT)
7. Install Jetson-compatible PyTorch wheel
8. Verify CUDA tensor execution

---

## CUDA Verification

```bash
nvcc --version
```

---

## TensorRT Verification

```bash
trtexec --help | head
```

## PyTorch CUDA Verification

```bash
python3 -c "import torch; x=torch.randn(1,3,224,224, device='cuda'); print(x.shape, x.device)"
```
Output:
```
torch.Size([1, 3, 224, 224]) cuda:0
```

## TensorRT Verification

```
docs/               Setup documentation
scripts/            Verification scripts
troubleshooting/    Issues encountered during setup
```

