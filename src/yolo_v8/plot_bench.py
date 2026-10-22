"""runs/bench_*.log 를 읽어 전력 모드별 벤치마크 막대그래프(SVG)를 만든다.

외부 라이브러리 없이 표준 라이브러리만 사용한다.

사용 예 (저장소 루트에서):
  python src/yolo_v8/plot_bench.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = {"15W": ROOT / "runs" / "bench_15w.log",
        "MAXN_SUPER": ROOT / "runs" / "bench_maxn_super.log"}
OUT = Path(__file__).resolve().parent / "assets" / "bench_power_mode.svg"

LINE = re.compile(r"\[(yolov8n\.\w+)\] 전체 ([\d.]+) ms \(([\d.]+) fps\) \| inference ([\d.]+) ms")

# 팔레트 슬롯 1(파랑), 2(주황) / 표면·텍스트 토큰
SURFACE, GRID = "#fcfcfb", "#e4e3df"
INK, INK2 = "#0b0b0b", "#52514e"
SERIES = [("yolov8n.pt", "PyTorch (FP16)", "#2a78d6"),
          ("yolov8n.engine", "TensorRT engine", "#eb6834")]


def parse(path):
    res = {}
    for m in LINE.finditer(path.read_text(encoding="utf-8")):
        res[m.group(1)] = {"total": float(m.group(2)), "fps": float(m.group(3)),
                           "infer": float(m.group(4))}
    return res


def bar(x, base, w, h, color, r=4):
    """밑은 사각, 끝(위)만 4px 둥근 막대."""
    y = base - h
    d = f"M{x},{base} V{y + r} a{r},{r} 0 0 1 {r},{-r} H{x + w - r} a{r},{r} 0 0 1 {r},{r} V{base} Z"
    return f'<path d="{d}" fill="{color}"/>'


def panel(px, title, key, data, ymax):
    top, base, w = 96, 280, 24
    sx0, sx1 = px + 44, px + 316
    out = [f'<text x="{px}" y="72" font-size="14" font-weight="600" fill="{INK}">{title}</text>']
    for t in range(0, ymax + 1, 10):
        y = base - (base - top) * t / ymax
        out.append(f'<line x1="{sx0}" y1="{y}" x2="{sx1}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{sx0 - 8}" y="{y + 4}" font-size="11" text-anchor="end" fill="{INK2}">{t}</text>')
    modes = list(LOGS)
    for i, mode in enumerate(modes):
        cx = sx0 + (sx1 - sx0) * (i + 0.5) / len(modes)
        for j, (model, _, color) in enumerate(SERIES):
            v = data[mode][model][key]
            h = (base - top) * v / ymax
            x = cx - w - 1 + j * (w + 2)  # 막대 사이 2px 표면 간격
            out.append(bar(x, base, w, h, color))
            out.append(f'<text x="{x + w / 2}" y="{base - h - 6}" font-size="11" '
                       f'text-anchor="middle" fill="{INK}">{v:.1f}</text>')
        out.append(f'<text x="{cx}" y="{base + 20}" font-size="12" text-anchor="middle" fill="{INK2}">{mode}</text>')
    out.append(f'<line x1="{sx0}" y1="{base}" x2="{sx1}" y2="{base}" stroke="{GRID}" stroke-width="1"/>')
    return "\n".join(out)


def main():
    data = {mode: parse(p) for mode, p in LOGS.items()}
    ymax = 40  # 두 패널이 같은 y축(ms)을 쓰도록 맞춤
    legend = []
    for i, (_, label, color) in enumerate(SERIES):
        x = 32 + i * 150
        legend.append(f'<rect x="{x}" y="46" width="12" height="12" rx="2" fill="{color}"/>'
                      f'<text x="{x + 18}" y="57" font-size="12" fill="{INK2}">{label}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 330" width="720" height="330" \
font-family="-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">
<title id="t">YOLOv8n latency by power mode</title>
<desc id="d">Grouped bars of end-to-end and inference-only latency in ms (lower is better) for PyTorch FP16 and TensorRT under 15W and MAXN_SUPER.</desc>
<rect width="720" height="330" rx="8" fill="{SURFACE}"/>
<text x="32" y="30" font-size="16" font-weight="600" fill="{INK}">YOLOv8n latency: 15W vs MAXN_SUPER (ms, lower is better)</text>
{''.join(legend)}
{panel(32, "End-to-end", "total", data, ymax)}
{panel(372, "Inference only", "infer", data, ymax)}
</svg>
'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
