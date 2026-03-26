import time
from collections import defaultdict

import torch
from ultralytics import YOLO


def sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ModuleProfiler:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.start_times = {}
        self.module_times = defaultdict(list)

    def _pre_hook(self, name):
        def hook(module, inputs):
            sync_if_needed()
            self.start_times[name] = time.perf_counter()
        return hook

    def _post_hook(self, name):
        def hook(module, inputs, output):
            sync_if_needed()
            end = time.perf_counter()
            start = self.start_times.pop(name, None)
            if start is not None:
                self.module_times[name].append((end - start) * 1000.0)
        return hook

    def register(self):
        for name, module in self.model.named_modules():
            # leaf module만 profiling
            if len(list(module.children())) == 0:
                self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
                self.handles.append(module.register_forward_hook(self._post_hook(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def summary(self, topk=30):
        rows = []
        for name, times in self.module_times.items():
            if len(times) == 0:
                continue
            total = sum(times)
            avg = total / len(times)
            rows.append((name, avg, total, len(times)))

        rows.sort(key=lambda x: x[2], reverse=True)

        print("\n=== Module-wise Profiling Result (sorted by total time) ===")
        print(f"{'module':60s} {'avg_ms':>10s} {'total_ms':>12s} {'calls':>8s}")
        print("-" * 95)
        for name, avg, total, count in rows[:topk]:
            print(f"{name:60s} {avg:10.3f} {total:12.3f} {count:8d}")


def main():
    model_wrapper = YOLO("yolov8n.pt")
    model = model_wrapper.model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    x = torch.randn(1, 3, 640, 640, device=device)

    profiler = ModuleProfiler(model)
    profiler.register()

    warmup_iters = 10
    measure_iters = 20

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)

        for i in range(measure_iters):
            _ = model(x)
            print(f"[{i+1:02d}/{measure_iters}] done")

    profiler.remove()
    profiler.summary(topk=50)


if __name__ == "__main__":
    main()