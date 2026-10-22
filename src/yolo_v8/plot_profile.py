"""runs/bench_maxn_super_*.log (2~9번 스크립트, trtexec 결과)를 읽어 SVG 차트 4개를 만든다.

표준 라이브러리만 사용한다. 도형 헬퍼와 색은 plot_bench.py 를 그대로 쓴다.

사용 예 (저장소 루트에서):
  python src/yolo_v8/plot_profile.py
"""
import re
from pathlib import Path

from plot_bench import GRID, INK, INK2, SURFACE

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
ASSETS = Path(__file__).resolve().parent / "assets"
PFX = "bench_maxn_super_"

# 팔레트: 백엔드(파랑/aqua/주황), 단계(magenta/green/violet) - 색은 항목을 따라간다
PYTORCH, ONNX, TRT = "#2a78d6", "#1baf7a", "#eb6834"
PRE, INF, POST = "#e87ba4", "#008300", "#6250d6"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def read(name):
    return (RUNS / f"{PFX}{name}.log").read_text(encoding="utf-8")


def f(pattern, text):
    return float(re.search(pattern, text).group(1))


# ---- 로그 파싱 -------------------------------------------------------------
def parse_all():
    d = {}
    t = read("2_multiple_whole")
    d["pt_full"] = f(r"avg latency : ([\d.]+) ms", t)
    t = read("4_forward_only")
    d["pt_fwd"] = f(r"forward\s+avg=\s*([\d.]+) ms", t)
    t = read("3_3stage_separate")
    d["stages_pt"] = [f(rf"{s}\s+avg=\s*([\d.]+) ms", t) for s in ("preprocess", "inference", "postprocess")]
    for key, name in (("onnx", "7_onnx_profile"), ("trt_py", "9_tensorrt_inference")):
        t = read(name)
        d[key] = {s: f(rf"{s}\s+avg=\s*([\d.]+) ms", t) for s in ("total", "preprocess", "inference")}
    t = read("trtexec")
    d["trtexec"] = f(r"Latency: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms", t)
    t = read("5_module_separate")
    mods = re.findall(r"^(model\.\S+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)$", t, re.M)
    d["modules"] = [(n, float(a), float(tot), int(c)) for n, a, tot, c in mods]
    t = read("6_profiler")
    ops = re.findall(r"^(aten::\S+)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)$", t, re.M)
    d["ops"] = [(n, float(ms), float(r), int(c)) for n, ms, r, c in ops]
    return d


# ---- SVG 헬퍼 ---------------------------------------------------------------
def frame(title, subtitle, height, body, legend=()):
    leg = "".join(
        f'<rect x="{32 + i * 150}" y="{46}" width="12" height="12" rx="2" fill="{c}"/>'
        f'<text x="{50 + i * 150}" y="57" font-size="12" fill="{INK2}">{lbl}</text>'
        for i, (lbl, c) in enumerate(legend))
    sub = f'<text x="32" y="{44 if not legend else 76}" font-size="12" fill="{INK2}">{subtitle}</text>' if subtitle else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 {height}" width="720" height="{height}" \
font-family="{FONT}" role="img" aria-label="{title}">
<title>{title}</title>
<rect width="720" height="{height}" rx="8" fill="{SURFACE}"/>
<text x="32" y="30" font-size="16" font-weight="600" fill="{INK}">{title}</text>
{leg}{sub}
{body}
</svg>
'''


def hpath(x, y, w, h, r=4, round_end=True):
    """왼쪽(기준선)은 사각, 오른쪽 끝만 4px 둥근 가로 막대."""
    if not round_end or w <= r * 2:
        return f"M{x},{y} h{w} v{h} h{-w} Z"
    return f"M{x},{y} H{x + w - r} a{r},{r} 0 0 1 {r},{r} V{y + h - r} a{r},{r} 0 0 1 {-r},{r} H{x} Z"


def luminance(hexcolor):
    rgb = [int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def grid(x0, x1, top, bottom, xmax, ticks, unit=""):
    out = []
    for t in ticks:
        x = x0 + (x1 - x0) * t / xmax
        out.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x}" y="{bottom + 16}" font-size="11" text-anchor="middle" fill="{INK2}">{t}{unit}</text>')
    return "".join(out)


def hbars(title, subtitle, rows, xmax, ticks, label_w=210, unit="", legend=(), thick=20, pitch=44):
    """rows: (라벨, 보조라벨, 값, 색, 값 텍스트)"""
    top = 96 if legend else 64
    x0, x1 = 32 + label_w, 640
    bottom = top + pitch * len(rows)
    body = [grid(x0, x1, top - 8, bottom - 8, xmax, ticks, unit)]
    for i, (label, sub, v, color, vt) in enumerate(rows):
        y = top + i * pitch
        w = (x1 - x0) * v / xmax
        body.append(f'<text x="{x0 - 12}" y="{y + 11}" font-size="12" text-anchor="end" fill="{INK}">{label}</text>')
        if sub:
            body.append(f'<text x="{x0 - 12}" y="{y + 25}" font-size="10" text-anchor="end" fill="{INK2}">{sub}</text>')
        body.append(f'<path d="{hpath(x0, y, w, thick)}" fill="{color}"/>')
        body.append(f'<text x="{x0 + w + 8}" y="{y + 14}" font-size="12" fill="{INK}">{vt}</text>')
    return frame(title, subtitle, bottom + 30, "\n".join(body), legend)


# ---- 차트 -------------------------------------------------------------------
def chart_backends(d):
    rows = [
        ("ONNX Runtime (CPU)", "preprocess + inference", d["onnx"]["total"], ONNX),
        ("PyTorch, full pipeline", "pre + inference + post", d["pt_full"], PYTORCH),
        ("PyTorch, forward only", "model forward", d["pt_fwd"], PYTORCH),
        ("TensorRT FP16, Python", "preprocess + inference", d["trt_py"]["total"], TRT),
        ("TensorRT FP16, trtexec", "H2D + GPU + D2H", d["trtexec"], TRT),
    ]
    rows = [(a, b, v, c, f"{v:.1f} ms") for a, b, v, c in rows]
    return hbars("YOLOv8n latency by backend (MAXN_SUPER)", "Average latency, lower is better. Each script measures a different scope.",
                 rows, 150, (0, 50, 100, 150), legend=(("PyTorch", PYTORCH), ("ONNX Runtime", ONNX), ("TensorRT", TRT)))


def chart_stages(d):
    pre, inf, post = d["stages_pt"]
    rows = [("PyTorch", [pre, inf, post]),
            ("ONNX Runtime (CPU)", [d["onnx"]["preprocess"], d["onnx"]["inference"], 0]),
            ("TensorRT FP16, Python", [d["trt_py"]["preprocess"], d["trt_py"]["inference"], 0])]
    colors = (PRE, INF, POST)
    xmax, x0, x1, top, pitch, thick = 150, 190, 640, 96, 44, 20
    scale = (x1 - x0) / xmax
    body = [grid(x0, x1, top - 8, top + pitch * len(rows) - 8, xmax, (0, 50, 100, 150))]
    for i, (name, vals) in enumerate(rows):
        y = top + i * pitch
        body.append(f'<text x="{x0 - 12}" y="{y + 14}" font-size="12" text-anchor="end" fill="{INK}">{name}</text>')
        segs = [(v, c) for v, c in zip(vals, colors) if v > 0]
        x = x0
        for j, (v, c) in enumerate(segs):
            w = v * scale
            last = j == len(segs) - 1
            body.append(f'<path d="{hpath(x, y, w, thick, round_end=last)}" fill="{c}"/>')
            label = f"{v:.1f}"
            if w > 46:  # 글자가 들어갈 만큼 넓을 때만 안쪽에 표기
                fill = INK if luminance(c) > 0.3 else "#ffffff"
                body.append(f'<text x="{x + w / 2}" y="{y + 14}" font-size="11" text-anchor="middle" fill="{fill}">{label}</text>')
            x += w + 2
        body.append(f'<text x="{x + 6}" y="{y + 14}" font-size="12" fill="{INK}">{sum(vals):.1f} ms</text>')
    return frame("Where the time goes: stage breakdown (MAXN_SUPER)", "Average ms per image. Segment labels shown only where they fit.",
                 top + pitch * len(rows) + 30, "\n".join(body),
                 legend=(("Preprocess", PRE), ("Inference", INF), ("Postprocess", POST)))


def chart_modules(d):
    top10 = d["modules"][:10]
    rows = [(n, f"{c} calls, {a:.2f} ms avg", tot, PYTORCH, f"{tot:.0f} ms") for n, a, tot, c in top10]
    return hbars("Slowest modules, PyTorch (MAXN_SUPER)", "Total time over 30 forward passes (10 warm-up + 20 measured), top 10.",
                 rows, 300, (0, 100, 200, 300), label_w=200, thick=18, pitch=40)


def chart_ops(d):
    top10 = d["ops"][:10]
    rows = [(n, f"{c} calls, {ms:.0f} ms", r, PYTORCH, f"{r:.1f}%") for n, ms, r, c in top10]
    return hbars("Slowest ops, PyTorch profiler (MAXN_SUPER)", "Share of summed CPU time, top 10. Ops are nested (conv2d contains convolution), so shares overlap.",
                 rows, 16, (0, 5, 10, 15), label_w=210, unit="%", thick=18, pitch=40)


def main():
    d = parse_all()
    ASSETS.mkdir(exist_ok=True)
    for name, fn in (("backends_latency", chart_backends), ("stage_breakdown", chart_stages),
                     ("module_profile", chart_modules), ("op_profile", chart_ops)):
        out = ASSETS / f"{name}.svg"
        out.write_text(fn(d), encoding="utf-8")
        print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
