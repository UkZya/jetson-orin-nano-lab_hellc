from pathlib import Path

from ultralytics import YOLO
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트


def main():
    model = YOLO("yolov8n.pt")
    results = model(str(ROOT / "bus.jpg"))

    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(annotated_rgb)
    img.save(ROOT / "bus_result.jpg")

    print("saved to bus_result.jpg")


if __name__ == "__main__":
    main()