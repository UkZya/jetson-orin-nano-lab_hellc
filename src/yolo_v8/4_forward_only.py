import time
from statistics import mean

import torch
from ultralytics import YOLO


def sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(name, values_ms):
    avg_ms = mean(values_ms)
    min_ms = min(values_ms)
    max_ms = max(values_ms)

    print(f"{name:<12} avg={avg_ms:7.2f} ms | min={min_ms:7.2f} ms | max={max_ms:7.2f} ms")
    return avg_ms


def main():
    model_wrapper = YOLO("yolov8n.pt")
    model = model_wrapper.model
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch_size = 1
    height = 640
    width = 640

    # 이미 전처리 완료된 입력 tensor라고 가정
    x = torch.randn(batch_size, 3, height, width, device=device)

    warmup_iters = 20
    measure_iters = 100

    print(f"[INFO] Device        : {device}")
    print(f"[INFO] Input shape   : {tuple(x.shape)}")
    print(f"[INFO] Warmup iters  : {warmup_iters}")
    print(f"[INFO] Measure iters : {measure_iters}")

    # warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            sync_if_needed()
            _ = model(x)
            sync_if_needed()

    latencies_ms = []

    with torch.no_grad():
        for i in range(measure_iters):
            sync_if_needed()
            t0 = time.perf_counter()

            y = model(x)

            sync_if_needed()
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            latencies_ms.append(latency_ms)

            print(f"[{i+1:03d}/{measure_iters}] forward = {latency_ms:.2f} ms")

    print("\n=== Forward-only Benchmark Result ===")
    avg_ms = summarize("forward", latencies_ms)
    print(f"fps         : {1000.0 / avg_ms:.2f}")

    # 출력 shape 대충 확인
    if isinstance(y, (list, tuple)):
        print("[INFO] Output type  : list/tuple")
        for idx, out in enumerate(y):
            if hasattr(out, "shape"):
                print(f"[INFO] Output[{idx}] shape: {tuple(out.shape)}")
            else:
                print(f"[INFO] Output[{idx}] type : {type(out)}")
    else:
        if hasattr(y, "shape"):
            print(f"[INFO] Output shape : {tuple(y.shape)}")
        else:
            print(f"[INFO] Output type  : {type(y)}")


if __name__ == "__main__":
    main()