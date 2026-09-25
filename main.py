from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# -----------------------------
# Пути
# -----------------------------

FRAMES_DIR = Path("frames")
DETECTION_DIR = Path("detection")
SEGMENTATION_DIR = Path("segmentation")

DETECTION_DIR.mkdir(exist_ok=True)
SEGMENTATION_DIR.mkdir(exist_ok=True)


# -----------------------------
# Загружаем модели
# -----------------------------

# Модель для обычной детекции
detection_model = YOLO("yolo11n.pt")

# Модель для сегментации
segmentation_model = YOLO("yolo11n-seg.pt")


# -----------------------------
# Получаем список кадров
# -----------------------------

frames = sorted(FRAMES_DIR.glob("*.jpg"))

print(f"Найдено кадров: {len(frames)}")


# -----------------------------
# Детекция
# -----------------------------

print("Начинаем детекцию...")

for frame_path in frames:

    results = detection_model(
        str(frame_path),
        verbose=False
    )

    result = results[0]

    # Рисуем bounding boxes
    annotated = result.plot()

    output_path = DETECTION_DIR / frame_path.name

    cv2.imwrite(
        str(output_path),
        annotated
    )

print("Детекция завершена.")


# -----------------------------
# Сегментация
# -----------------------------

print("Начинаем сегментацию...")

for frame_path in frames:

    results = segmentation_model(
        str(frame_path),
        verbose=False
    )

    result = results[0]

    # Результат с масками
    annotated = result.plot(
        boxes=True,
        labels=True,
        masks=True
    )

    output_path = SEGMENTATION_DIR / frame_path.name

    cv2.imwrite(
        str(output_path),
        annotated
    )

print("Сегментация завершена.")


# -----------------------------
# Функция создания GIF
# -----------------------------

def create_gif(input_dir, output_file, duration=200):

    images = []

    image_files = sorted(
        Path(input_dir).glob("*.jpg")
    )

    for image_path in image_files:

        image = Image.open(image_path).convert("RGB")

        images.append(image)

    if not images:
        print(f"Нет изображений в {input_dir}")
        return

    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF создан: {output_file}")


# -----------------------------
# Создание GIF
# -----------------------------

create_gif(
    DETECTION_DIR,
    "detection.gif"
)

create_gif(
    SEGMENTATION_DIR,
    "segmentation.gif"
)


print("Готово!")