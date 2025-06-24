# balance_classes.py
"""
Balancea las clases en un dataset de imágenes, igualando la cantidad de imágenes en cada clase
a la clase minoritaria (undersampling) o a la clase mayoritaria (oversampling, opcional).
Por defecto, realiza undersampling (elimina aleatoriamente imágenes de las clases con más archivos).
"""

import os
import random
from pathlib import Path


def balance_classes(data_dir, mode="undersample", seed=42):
    """
    data_dir: carpeta con subcarpetas de clases
    mode: 'undersample' (default) o 'oversample'
    """
    random.seed(seed)
    data_dir = Path(data_dir)
    class_counts = {}
    class_files = {}
    # Contar archivos por clase
    for class_folder in data_dir.iterdir():
        if not class_folder.is_dir():
            continue
        files = sorted([f for f in class_folder.glob("*.jpg")])
        class_counts[class_folder.name] = len(files)
        class_files[class_folder.name] = files
    print("Conteo actual por clase:", class_counts)
    if mode == "undersample":
        min_count = min(class_counts.values())
        print(f"Reduciendo todas las clases a {min_count} imágenes...")
        for class_name, files in class_files.items():
            if len(files) > min_count:
                to_remove = random.sample(files, len(files) - min_count)
                for f in to_remove:
                    os.remove(f)
                    print(f"Eliminado: {f}")
    elif mode == "oversample":
        max_count = max(class_counts.values())
        print(f"Aumentando todas las clases a {max_count} imágenes...")
        for class_name, files in class_files.items():
            if len(files) < max_count:
                needed = max_count - len(files)
                for i in range(needed):
                    src = random.choice(files)
                    dst = src.parent / f"{src.stem}_aug{i}{src.suffix}"
                    os.link(src, dst)  # Copia dura para ahorrar espacio
                    print(f"Copiado: {src} -> {dst}")
    else:
        print("Modo no soportado. Usa 'undersample' u 'oversample'.")


if __name__ == "__main__":
    DATA_DIR = "total/archive/Augmented"  # Cambia a tu directorio de datos
    balance_classes(DATA_DIR, mode="undersample")
