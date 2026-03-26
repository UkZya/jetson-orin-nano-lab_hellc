from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    output_path = model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
    )

    print(f"[INFO] Exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()