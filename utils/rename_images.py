# rename_images.py
"""
Renombra imágenes en total/archive/Training/**/**.jpg con el formato:
primera letra de la carpeta + _ + número incremental único por carpeta (ej: g_1.jpg, m_1.jpg)
"""

import os
import glob


def rename_images(base_dir):
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        prefix = class_dir[0].lower()
        images = sorted(glob.glob(os.path.join(class_path, "*.jpg")))
        num_digits = 5
        for idx, img_path in enumerate(images, 1):
            ext = os.path.splitext(img_path)[1]
            new_name = f"{prefix}_{str(idx).zfill(num_digits)}{ext}"
            new_path = os.path.join(class_path, new_name)
            if os.path.abspath(img_path) != os.path.abspath(new_path):
                os.rename(img_path, new_path)
                print(f"Renombrado: {img_path} -> {new_path}")


if __name__ == "__main__":
    BASE_DIR = "total/archive/Training"
    rename_images(BASE_DIR)
