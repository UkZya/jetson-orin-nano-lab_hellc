#!/bin/bash

echo "=============================="
echo "Jetson System Information"
echo "=============================="

echo
echo "Hostname:"
hostname

echo
echo "Jetson Linux Version:"
cat /etc/nv_tegra_release

echo
echo "Kernel:"
uname -a

echo
echo "CPU Info:"
lscpu | grep "Model name"

echo
echo "Memory:"
free -h

echo
echo "Disk:"
df -h /

echo
echo "CUDA Version:"
nvcc --version | grep release

echo
echo "TensorRT Version:"
dpkg -l | grep nvinfer | head

echo
echo "GPU Info (nvidia-smi):"
nvidia-smi

echo
echo "Jetson GPU Stats:"
sudo tegrastats --interval 1000 --count 1 2>/dev/null

echo
echo "PyTorch:"
python3 - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF