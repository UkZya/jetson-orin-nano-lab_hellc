import time
from statistics import mean

import cv2
import torch
from ultralytics import YOLO


IMAGE_PATH = "/home/namuk/projects/jetson-orin-nano-lab/bus.jpg"


def sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(name, values_ms, total_avg_ms=None):
    avg_ms = mean(values_ms)
    min_ms = min(values_ms)
    max_ms = max(values_ms)

    print(f"{name:<12} avg={avg_ms:7.2f} ms | min={min_ms:7.2f} ms | max={max_ms:7.2f} ms", end="")
    if total_avg_ms is not None:
        ratio = (avg_ms / total_avg_ms) * 100.0
        print(f" | ratio={ratio:6.2f}%")
    else:
        print()

    return avg_ms


def main():
    model = YOLO("yolov8n.pt")

    # 원본 이미지는 한 번만 읽음 (디스크 I/O 제외)
    im0 = cv2.imread(IMAGE_PATH)
    if im0 is None:
        raise FileNotFoundError(f"Failed to read image: {IMAGE_PATH}")

    # predictor 초기화용 1회 실행
    _ = model.predict(im0, imgsz=640, verbose=False, save=False)
    predictor = model.predictor

    warmup_iters = 10
    measure_iters = 50

    preprocess_ms = []
    inference_ms = []
    postprocess_ms = []
    total_ms = []

    print(f"[INFO] Warmup: {warmup_iters} iterations")
    for _ in range(warmup_iters):
        im0s = [im0]

        sync_if_needed()
        im = predictor.preprocess(im0s)

        sync_if_needed()
        preds = predictor.inference(im)

        sync_if_needed()
        _ = predictor.postprocess(preds, im, im0s)

        sync_if_needed()

    print(f"[INFO] Measure: {measure_iters} iterations")
    for i in range(measure_iters):
        im0s = [im0]

        sync_if_needed()
        t0 = time.perf_counter()

        im = predictor.preprocess(im0s)
        sync_if_needed()
        t1 = time.perf_counter()

        preds = predictor.inference(im)
        sync_if_needed()
        t2 = time.perf_counter()

        results = predictor.postprocess(preds, im, im0s)
        sync_if_needed()
        t3 = time.perf_counter()

        pre = (t1 - t0) * 1000.0
        inf = (t2 - t1) * 1000.0
        post = (t3 - t2) * 1000.0
        total = (t3 - t0) * 1000.0

        preprocess_ms.append(pre)
        inference_ms.append(inf)
        postprocess_ms.append(post)
        total_ms.append(total)

        print(
            f"[{i+1:03d}/{measure_iters}] "
            f"pre={pre:6.2f} ms | "
            f"inf={inf:6.2f} ms | "
            f"post={post:6.2f} ms | "
            f"total={total:6.2f} ms"
        )

    print("\n=== Stage-wise Benchmark Result ===")
    total_avg = mean(total_ms)

    summarize("preprocess", preprocess_ms, total_avg)
    summarize("inference", inference_ms, total_avg)
    summarize("postprocess", postprocess_ms, total_avg)
    summarize("total", total_ms)

    print(f"\nEstimated FPS (total avg): {1000.0 / total_avg:.2f}")

    # 마지막 결과 확인용
    print("\n[INFO] Last result summary:")
    print(results[0])


if __name__ == "__main__":
    main()