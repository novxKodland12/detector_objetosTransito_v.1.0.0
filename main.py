import argparse
from pathlib import Path
import random

import cv2
from ultralytics import YOLO


# CLASES EN INGLÉS (COCO)
DEFAULT_CLASSES = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "traffic light", "stop sign",
    "dog", "cat"
]


COLORS = {
    "person": (0, 0, 255),   # rojo
    "bicycle": (0, 255, 0),  # verde
    "car": (255, 0, 0),      # azul
    "motorcycle": (0, 255, 255),
    "bus": (0, 165, 255),
    "truck": (128, 0, 128),
    "traffic light": (255, 255, 0),
    "stop sign": (255, 0, 255),
    "dog": (19, 69, 139),
    "cat": (128, 128, 128)
}


def get_class_colors(classes):
    return {cls: generate_color() for cls in classes}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 detección + recortes + imagen anotada"
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--input", default="demostracion.jpeg")
    parser.add_argument("--output-dir", default="recortes")
    parser.add_argument("--annotated", default="resultado.jpg")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser.parse_args()


def draw_label(img, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - h - 10), (x + w, y), color, -1)
    cv2.putText(
        img, text, (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 0), 2, cv2.LINE_AA
    )


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    annotated_path = Path(args.annotated)

    if not input_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {input_path}")

    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError("No se pudo cargar la imagen")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # 🔥 VALIDAR CLASES REALES DEL MODELO
    model_classes = {v.lower() for v in model.names.values()}
    selected_classes = [c.lower() for c in args.classes if c.lower() in model_classes]

    if not selected_classes:
        raise ValueError("Ninguna clase válida. Usa nombres en inglés del modelo.")

    results = model(image, verbose=False)

    annotated = image.copy()

    colors = COLORS

    saved_count = 0
    class_counter = {cls: 0 for cls in selected_classes}

    for result in results:
        for box in result.boxes:

            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, str(cls_id)).lower()

            if cls_name not in selected_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            filename = f"{cls_name}_{saved_count:03d}.jpg"
            cv2.imwrite(str(output_dir / filename), crop)

            class_counter[cls_name] += 1

            color = colors.get(cls_name, (0, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            conf = float(box.conf[0]) if hasattr(box, "conf") else 0
            label = f"{cls_name} {conf:.2f}"
            draw_label(annotated, label, x1, y1, color)

            saved_count += 1

    cv2.imwrite(str(annotated_path), annotated)

    print("\n📊 RESUMEN:")
    for cls, count in class_counter.items():
        if count > 0:
            print(f" - {cls}: {count}")

    print(f"\n✅ Recortes: {saved_count}")
    print(f"📁 Carpeta: {output_dir}")
    print(f"🖼️ Imagen final: {annotated_path}")


if __name__ == "__main__":
    main()
