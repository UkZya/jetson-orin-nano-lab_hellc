import time
from statistics import mean

import cv2
import numpy as np
import tensorrt as trt
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


IMAGE_PATH = "/home/namuk/projects/jetson-orin-nano-lab/bus.jpg"
ENGINE_PATH = "/home/namuk/projects/jetson-orin-nano-lab/engines/yolov8n_fp16.engine"


def check_cuda(status):
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Runtime Error: {status}")


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
    im = np.transpose(im, (2, 0, 1))   # HWC -> CHW
    im = np.expand_dims(im, axis=0)     # CHW -> BCHW
    im = np.ascontiguousarray(im)
    return im, ratio, dwdh


def summarize(name, values_ms):
    avg_ms = mean(values_ms)
    min_ms = min(values_ms)
    max_ms = max(values_ms)
    print(f"{name:<12} avg={avg_ms:7.2f} ms | min={min_ms:7.2f} ms | max={max_ms:7.2f} ms")
    return avg_ms


class TRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        # CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        check_cuda(err)

        self.input_name = None
        self.output_names = []
        self.host_buffers = {}
        self.device_buffers = {}
        self.tensor_shapes = {}
        self.tensor_dtypes = {}

        self._allocate_buffers()

    def _trt_dtype_to_np(self, dtype):
        return np.dtype(trt.nptype(dtype))

    def _allocate_buffers(self):
        num_tensors = self.engine.num_io_tensors

        for i in range(num_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self._trt_dtype_to_np(self.engine.get_tensor_dtype(name))

            self.tensor_shapes[name] = shape
            self.tensor_dtypes[name] = dtype

            size = int(np.prod(shape))
            host_array = np.empty(size, dtype=dtype)

            err, device_ptr = cudart.cudaMalloc(host_array.nbytes)
            check_cuda(err)

            self.host_buffers[name] = host_array
            self.device_buffers[name] = device_ptr

            # execute_async_v3 방식에서는 tensor address를 명시적으로 연결
            ok = self.context.set_tensor_address(name, int(device_ptr))
            if not ok:
                raise RuntimeError(f"Failed to set tensor address for {name}")

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if self.input_name is None:
            raise RuntimeError("No input tensor found")

    def infer(self, input_array):
        input_array = np.ascontiguousarray(input_array)
        expected_shape = self.tensor_shapes[self.input_name]
        expected_dtype = self.tensor_dtypes[self.input_name]

        if tuple(input_array.shape) != tuple(expected_shape):
            raise ValueError(
                f"Input shape mismatch. expected={expected_shape}, got={input_array.shape}"
            )
        if input_array.dtype != expected_dtype:
            input_array = input_array.astype(expected_dtype, copy=False)

        # H2D
        err = cudart.cudaMemcpyAsync(
            self.device_buffers[self.input_name],
            input_array.ctypes.data,
            input_array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )[0]
        check_cuda(err)

        # Execute
        ok = self.context.execute_async_v3(stream_handle=self.stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        # D2H
        outputs = {}
        for name in self.output_names:
            host_arr = self.host_buffers[name]
            err = cudart.cudaMemcpyAsync(
                host_arr.ctypes.data,
                self.device_buffers[name],
                host_arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )[0]
            check_cuda(err)

        err = cudart.cudaStreamSynchronize(self.stream)[0]
        check_cuda(err)

        for name in self.output_names:
            shape = self.tensor_shapes[name]
            outputs[name] = self.host_buffers[name].reshape(shape).copy()

        return outputs

    def destroy(self):
        for _, ptr in self.device_buffers.items():
            cudart.cudaFree(ptr)
        if self.stream is not None:
            cudart.cudaStreamDestroy(self.stream)


def main():
    im0 = cv2.imread(IMAGE_PATH)
    if im0 is None:
        raise FileNotFoundError(f"Failed to read image: {IMAGE_PATH}")

    trt_infer = TRTInfer(ENGINE_PATH)

    print(f"[INFO] Input tensor name : {trt_infer.input_name}")
    print(f"[INFO] Input tensor shape: {trt_infer.tensor_shapes[trt_infer.input_name]}")
    print(f"[INFO] Output tensors    : {trt_infer.output_names}")
    for name in trt_infer.output_names:
        print(f"    - {name}: shape={trt_infer.tensor_shapes[name]}, dtype={trt_infer.tensor_dtypes[name]}")

    warmup_iters = 20
    measure_iters = 100

    preprocess_ms = []
    inference_ms = []
    total_ms = []

    print(f"[INFO] Warmup: {warmup_iters} iterations")
    for _ in range(warmup_iters):
        x, _, _ = preprocess_bgr(im0, imgsz=640)
        _ = trt_infer.infer(x)

    print(f"[INFO] Measure: {measure_iters} iterations")
    last_outputs = None

    for i in range(measure_iters):
        t0 = time.perf_counter()
        x, ratio, dwdh = preprocess_bgr(im0, imgsz=640)
        t1 = time.perf_counter()

        outputs = trt_infer.infer(x)
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

    print("\n=== TensorRT FP16 Benchmark Result ===")
    avg_total = summarize("total", total_ms)
    summarize("preprocess", preprocess_ms)
    summarize("inference", inference_ms)
    print(f"fps         : {1000.0 / avg_total:.2f}")

    print("\n[INFO] Output summary:")
    for name, out in last_outputs.items():
        print(f"  {name}: shape={out.shape}, dtype={out.dtype}")

    trt_infer.destroy()


if __name__ == "__main__":
    main()