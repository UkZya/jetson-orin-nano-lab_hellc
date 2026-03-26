import torch
from ultralytics import YOLO
from torch.profiler import profile, ProfilerActivity


def get_time_value(event):
    # 버전별 profiler 속성 차이 대응
    if hasattr(event, "cuda_time_total"):
        return event.cuda_time_total, "CUDA"
    if hasattr(event, "self_cuda_time_total"):
        return event.self_cuda_time_total, "CUDA"
    if hasattr(event, "cpu_time_total"):
        return event.cpu_time_total, "CPU"
    return 0.0, "UNKNOWN"


def pretty_print(prof, topk=15):
    events = prof.key_averages()

    rows = []
    time_domain = None

    for e in events:
        t, domain = get_time_value(e)
        rows.append((e.key, t, e.count))
        if time_domain is None and domain != "UNKNOWN":
            time_domain = domain

    rows.sort(key=lambda x: x[1], reverse=True)
    total_time = sum(r[1] for r in rows)

    print(f"\n=== Pretty Op-wise Profiling (Top {topk}) ===")
    print(f"[INFO] Time domain: {time_domain}")
    print(f"{'op':35s} {'time(ms)':>12s} {'ratio':>10s} {'calls':>10s}")
    print("-" * 72)

    for key, t_us, count in rows[:topk]:
        t_ms = t_us / 1000.0  # us -> ms
        ratio = (t_us / total_time * 100.0) if total_time > 0 else 0.0
        print(f"{key:35s} {t_ms:12.3f} {ratio:9.2f}% {count:10d}")


def main():
    model_wrapper = YOLO("yolov8n.pt")
    model = model_wrapper.model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    x = torch.randn(1, 3, 640, 640, device=device)

    with torch.no_grad():
        # warmup
        for _ in range(10):
            _ = model(x)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            #acc_events=True,
        ) as prof:
            for _ in range(20):
                _ = model(x)

    pretty_print(prof, topk=15)


if __name__ == "__main__":
    main()