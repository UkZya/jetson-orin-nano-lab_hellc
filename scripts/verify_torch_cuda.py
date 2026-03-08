import torch

def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    x = torch.randn(1, 3, 224, 224, device="cuda")

    print("Tensor shape:", x.shape)
    print("Tensor device:", x.device)

if __name__ == "__main__":
    main()