import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    model = SimpleCNN().eval().cuda()
    dummy = torch.randn(1, 3, 224, 224, device="cuda")

    with torch.no_grad():
        y = model(dummy)
        print("PyTorch output shape:", y.shape)

    torch.onnx.export(
        model,
        dummy,
        "models/simple_cnn.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )
    print("Exported ONNX to models/simple_cnn.onnx")


if __name__ == "__main__":
    main()