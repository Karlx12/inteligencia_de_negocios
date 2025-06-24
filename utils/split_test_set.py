# split_test_set.py
"""
Este script separa aleatoriamente un 5% de las imágenes de cada clase en total/archive/Training/ y las mueve a total/archive/Testing/ manteniendo la estructura de carpetas.
"""

import random
import shutil
from pathlib import Path


def split_test_set(base_dir, test_dir, test_ratio=0.05, seed=42):
    random.seed(seed)
    base_dir = Path(base_dir)
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    for class_folder in base_dir.iterdir():
        if not class_folder.is_dir():
            continue
        images = sorted([f for f in class_folder.glob("*.jpg")])
        n_test = max(1, int(len(images) * test_ratio))
        test_images = random.sample(images, n_test)
        class_test_dir = test_dir / class_folder.name
        class_test_dir.mkdir(parents=True, exist_ok=True)
        for img_path in test_images:
            dest_path = class_test_dir / img_path.name
            shutil.move(str(img_path), str(dest_path))
            print(f"Movido a test: {img_path} -> {dest_path}")


if __name__ == "__main__":
    BASE_DIR = "total/archive/Augmented"  # Cambia a tu directorio de entrenamiento
    TEST_DIR = "total/archive/Testing"
    split_test_set(BASE_DIR, TEST_DIR)
