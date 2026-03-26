import time
from statistics import mean

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_PATH = "/home/namuk/projects/jetson-orin-nano-lab/bus.jpg"
ONNX_PATH = "/home/namuk/projects/jetson-orin-nano-lab/yolov8n.onnx"


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def preprocess_bgr(im_bgr, imgsz=640):
    im, ratio, dwdh = letterbox(im_bgr, new_shape=(imgsz, imgsz))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))  # HWC -> CHW
    im = np.expand_dims(im, axis=0)   # CHW -> BCHW
    im = np.ascontiguousarray(im)
    return im, ratio, dwdh


def summarize(name, values_ms):
    avg_ms = mean(values_ms)
    min_ms = min(values_ms)
    max_ms = max(values_ms)
    print(f"{name:<12} avg={avg_ms:7.2f} ms | min={min_ms:7.2f} ms | max={max_ms:7.2f} ms")
    return avg_ms


def main():
    im0 = cv2.imread(IMAGE_PATH)
    if im0 is None:
        raise FileNotFoundError(f"Failed to read image: {IMAGE_PATH}")

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"[INFO] ONNX Runtime providers: {available}")
    print(f"[INFO] Using providers      : {providers}")

    sess = ort.InferenceSession(ONNX_PATH, providers=providers)

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    print(f"[INFO] Input name   : {input_name}")
    print(f"[INFO] Output names : {output_names}")
    print(f"[INFO] Input shape  : {sess.get_inputs()[0].shape}")

    warmup_iters = 10
    measure_iters = 50

    preprocess_ms = []
    inference_ms = []
    total_ms = []

    print(f"[INFO] Warmup: {warmup_iters} iterations")
    for _ in range(warmup_iters):
        x, _, _ = preprocess_bgr(im0, imgsz=640)
        _ = sess.run(output_names, {input_name: x})

    print(f"[INFO] Measure: {measure_iters} iterations")
    last_outputs = None

    for i in range(measure_iters):
        t0 = time.perf_counter()
        x, ratio, dwdh = preprocess_bgr(im0, imgsz=640)
        t1 = time.perf_counter()

        outputs = sess.run(output_names, {input_name: x})
        t2 = time.perf_counter()

        pre = (t1 - t0) * 1000.0
        inf = (t2 - t1) * 1000.0
        total = (t2 - t0) * 1000.0

        preprocess_ms.append(pre)
        inference_ms.append(inf)
        total_ms.append(total)

        last_outputs = outputs

        print(
            f"[{i+1:03d}/{measure_iters}] "
            f"pre={pre:6.2f} ms | inf={inf:6.2f} ms | total={total:6.2f} ms"
        )

    print("\n=== ONNX Runtime Benchmark Result ===")
    avg_total = summarize("total", total_ms)
    summarize("preprocess", preprocess_ms)
    summarize("inference", inference_ms)
    print(f"fps         : {1000.0 / avg_total:.2f}")

    print("\n[INFO] Output summary:")
    for idx, out in enumerate(last_outputs):
        print(f"  output[{idx}] shape={out.shape}, dtype={out.dtype}")


if __name__ == "__main__":
    main()