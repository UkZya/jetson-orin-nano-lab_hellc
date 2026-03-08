# Jetson Orin Nano Lab

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
