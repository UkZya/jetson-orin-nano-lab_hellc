#!/bin/bash

echo "Checking CUDA compiler"
nvcc --version

echo
echo "Checking TensorRT"
trtexec --help | head

echo
echo "Checking GPU info"
sudo tegrastats 

