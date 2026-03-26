import time
from statistics import mean

from ultralytics import YOLO


IMAGE_PATH = "/home/namuk/projects/jetson-orin-nano-lab/bus.jpg"


def main():
    model = YOLO("yolov8n.pt")

    warmup_iters = 10
    measure_iters = 50

    print(f"[INFO] Warmup: {warmup_iters} iterations")
    for _ in range(warmup_iters):
        _ = model(IMAGE_PATH, verbose=False)

    latencies_ms = []

    print(f"[INFO] Measure: {measure_iters} iterations")
    for i in range(measure_iters):
        start = time.perf_counter()
        _ = model(IMAGE_PATH, verbose=False)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0
        latencies_ms.append(latency_ms)

        print(f"[{i+1:03d}/{measure_iters}] latency = {latency_ms:.2f} ms")

    print("\n=== Benchmark Result ===")
    print(f"avg latency : {mean(latencies_ms):.2f} ms")
    print(f"min latency : {min(latencies_ms):.2f} ms")
    print(f"max latency : {max(latencies_ms):.2f} ms")
    print(f"fps         : {1000.0 / mean(latencies_ms):.2f}")


if __name__ == "__main__":
    main()