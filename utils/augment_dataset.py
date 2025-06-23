# augment_dataset.py
"""
Aumenta significativamente el dataset de imágenes MRI usando las funciones de augmentación de utils.py.
Genera N aumentaciones por imagen y las guarda en una carpeta destino, manteniendo la estructura de clases.
"""

from pathlib import Path
from tqdm import tqdm
import cv2
import albumentations as A

AUG_PER_IMAGE = 10  # Número de aumentaciones por imagen
INPUT_DIR = "total/archive/Training"
OUTPUT_DIR = "total/archive/Augmented"

# Definir el pipeline de augmentación (basado en el notebook)
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CLAHE(p=0.3),
    ]
)


def augment_dataset(input_dir, output_dir, aug_per_image=AUG_PER_IMAGE):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        images = sorted([f for f in class_dir.glob("*.jpg")])
        for img_path in tqdm(images, desc=f"{class_dir.name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            # Guardar imagen original
            orig_name = out_class_dir / img_path.name
            cv2.imwrite(str(orig_name), img)
            # Generar aumentaciones
            for i in range(aug_per_image):
                augmented = transform(image=img)
                aug_img = augmented["image"]
                aug_name = out_class_dir / f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(aug_name), aug_img)


if __name__ == "__main__":
    augment_dataset(INPUT_DIR, OUTPUT_DIR)
