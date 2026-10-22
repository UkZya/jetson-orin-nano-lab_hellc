"""모델별 추론 속도 벤치마크 (영상 파일, 재생 속도 제한 없음).

사용 예:
  PYTHONPATH=~/trt_shim python src/bench_yolo.py yolov8n.pt yolov8n.engine --source videos/person-bicycle-car-detection.mp4
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("models", nargs="+")
p.add_argument("--source", required=True)
p.add_argument("--frames", type=int, default=300)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--imgsz", type=int, default=640)
args = p.parse_args()

cap = cv2.VideoCapture(args.source)
frames = []
while len(frames) < args.frames:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
print(f"프레임 {len(frames)}장 로드 ({frames[0].shape[1]}x{frames[0].shape[0]})")

for m in args.models:
    model = YOLO(m, task="detect")
    half = not m.endswith(".engine")  # 엔진은 변환 시 정밀도가 이미 결정됨
    kw = dict(imgsz=args.imgsz, device=0, verbose=False)
    if half:
        kw["half"] = True
    for f in frames[: args.warmup]:
        model.predict(f, **kw)

    total, inf = [], []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, **kw)[0]
        total.append((time.perf_counter() - t0) * 1000)
        inf.append(r.speed["inference"])
    total, inf = np.array(total), np.array(inf)
    print(f"[{m}] 전체 {total.mean():.1f} ms ({1000/total.mean():.1f} fps) | "
          f"inference {inf.mean():.1f} ms | p95 전체 {np.percentile(total, 95):.1f} ms")
