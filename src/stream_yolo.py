"""YOLO 추론 결과를 MJPEG 웹 스트림으로 내보낸다. (headless / SSH 포트 포워딩용)

사용 예:
  python src/stream_yolo.py --model models/yolov8n.pt --source videos/person-bicycle-car-detection.mp4
  브라우저: http://localhost:8080  (PC에서 ssh -L 8080:localhost:8080 ...)
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolov8n.pt")
p.add_argument("--source", required=True, help="영상 파일 경로 또는 카메라 번호(0)")
p.add_argument("--port", type=int, default=8080)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--fps", type=float, default=0, help="출력 fps 제한 (0이면 원본 fps)")
p.add_argument("--no-loop", action="store_true")
p.add_argument("--half", action="store_true", help="FP16 (pt 모델용, 엔진은 export 시 결정)")
args = p.parse_args()

latest = {"jpg": None}
cond = threading.Condition()


def worker():
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target = args.fps or src_fps
    model = YOLO(args.model, task="detect")
    print(f"모델 {args.model}, 소스 {args.source}, 목표 {target:.1f}fps", flush=True)

    infer_ms, t_prev, ema_fps, n = 0.0, time.time(), 0.0, 0
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            if args.no_loop:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        t1 = time.time()
        res = model.predict(frame, imgsz=args.imgsz, device=0, half=args.half, verbose=False)[0]
        infer_ms = (time.time() - t1) * 1000
        img = res.plot()

        n += 1
        now = time.time()
        inst = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        ema_fps = inst if n == 1 else 0.9 * ema_fps + 0.1 * inst
        label = f"FPS {ema_fps:4.1f} | infer {infer_ms:4.1f} ms | objects {len(res.boxes)}"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with cond:
            latest["jpg"] = buf.tobytes()
            cond.notify_all()

        # 영상 파일은 원본 속도로 재생되도록 제한
        sleep = 1.0 / target - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            html = b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%;max-width:960px'></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with cond:
                        cond.wait(timeout=5)
                        jpg = latest["jpg"]
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


threading.Thread(target=worker, daemon=True).start()
print(f"http://0.0.0.0:{args.port} 에서 스트리밍 중 (Ctrl+C로 종료)", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
