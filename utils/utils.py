"""
Utilidades para preprocesamiento de imágenes MRI y preparación para YOLOv7
"""

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # Add proper import for Rectangle
from tqdm import tqdm
import albumentations as A


# Funciones de preprocesamiento básico
def normalize_image(image):
    """Normaliza el brillo y contraste de una imagen"""
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    return (normalized * 255).astype(np.uint8)


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """Aplica Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Aplica desenfoque gaussiano para reducir ruido"""
    return cv2.GaussianBlur(image, kernel_size, 0)


def crop_brain_region(image, threshold=10):
    """Recorta la región cerebral de una imagen MRI"""
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y : y + h, x : x + w]
    return image


# Funciones para manejo de datos y directorios
def get_dataset_info(directory):
    """Obtiene información básica del dataset en el directorio especificado"""
    image_files = glob.glob(
        os.path.join(directory, "**", "*.jpg"), recursive=True
    ) + glob.glob(os.path.join(directory, "**", "*.png"), recursive=True)

    label_files = glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)

    # Filtrar posibles archivos no deseados como classes.txt
    label_files = [f for f in label_files if os.path.basename(f) != "classes.txt"]

    return {
        "total_images": len(image_files),
        "total_labels": len(label_files),
        "image_samples": image_files[:5],
        "label_samples": label_files[:5],
    }


# Pipeline de preprocesamiento
def preprocess_image(image_path, output_size=640):
    """
    Pipeline completo de preprocesamiento para una imagen MRI

    Args:
        image_path: Ruta de la imagen a preprocesar
        output_size: Tamaño de salida para la imagen

    Returns:
        Imagen preprocesada
    """
    # Cargar imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error cargando imagen: {image_path}")
        return None

    # 1. Normalización
    image = normalize_image(image)

    # 2. Recorte de región cerebral
    image = crop_brain_region(image)

    # 3. CLAHE para mejorar contraste
    image = apply_clahe(image)

    # 4. Suave desenfoque gaussiano para reducir ruido
    image = apply_gaussian_blur(image, kernel_size=(3, 3))

    # 5. Redimensionar a tamaño objetivo
    image = cv2.resize(image, (output_size, output_size))

    return image


def adjust_annotations(label_path, orig_shape, new_shape):
    """
    Ajusta las anotaciones YOLO basado en el cambio de tamaño
    """
    if not os.path.exists(label_path):
        return []

    orig_h, orig_w = orig_shape
    new_h, new_w = new_shape
    scale_x, scale_y = new_w / orig_w, new_h / orig_h

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()

        adjusted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)

                # Ajustar coordenadas basadas en el recorte y escala
                # Aplicar los factores de escala a las coordenadas
                x_center = x_center * scale_x
                y_center = y_center * scale_y
                width = width * scale_x
                height = height * scale_y

                adjusted_lines.append(
                    f"{int(class_id)} {x_center} {y_center} {width} {height}"
                )

        return adjusted_lines
    except Exception as e:
        print(f"Error al ajustar anotaciones {label_path}: {e}")
        return []


def process_dataset(input_dir, output_dir, target_size=640):
    """
    Procesa todas las imágenes y etiquetas en el directorio de entrada
    y las guarda en el directorio de salida
    """
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Encontrar todas las imágenes
    image_files = glob.glob(
        os.path.join(input_dir, "**", "*.jpg"), recursive=True
    ) + glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)

    print(f"Procesando {len(image_files)} imágenes...")

    for img_path in tqdm(image_files):
        # Obtener nombre base y ruta de etiqueta correspondiente
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(os.path.dirname(img_path), f"{base_name}.txt")

        # Cargar imagen original para obtener dimensiones
        orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None:
            print(f"Error cargando imagen: {img_path}")
            continue

        orig_shape = orig_img.shape

        # Preprocesar imagen
        processed_img = preprocess_image(img_path, output_size=target_size)
        if processed_img is None:
            continue

        # Ajustar anotaciones
        adjusted_annotations = adjust_annotations(
            label_path, orig_shape, (target_size, target_size)
        )

        # Guardar imagen procesada
        output_img_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_img_path, processed_img)

        # Guardar etiquetas ajustadas
        if adjusted_annotations:
            output_label_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(output_label_path, "w") as f:
                f.write("\n".join(adjusted_annotations))

    print(f"Procesamiento completado. Imágenes guardadas en {output_dir}")


def prepare_for_yolov7(image_path, label_path, target_size=640):
    """
    Prepara una imagen y sus anotaciones para el formato requerido por YOLOv7
    """
    # Cargar imagen y anotaciones
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Si es MRI en escala de grises
    h, w = image.shape

    # Redimensionar
    image = cv2.resize(image, (target_size, target_size))
    scale_x, scale_y = target_size / w, target_size / h

    # Procesar anotaciones (formato YOLO: class_id, x_center, y_center, width, height)
    with open(label_path, "r") as f:
        lines = f.readlines()
    yolo_annotations = []
    for line in lines:
        class_id, x, y, bw, bh = map(float, line.strip().split())
        x_center = x * scale_x / target_size
        y_center = y * scale_y / target_size
        width = bw * scale_x / target_size
        height = bh * scale_y / target_size
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return image, yolo_annotations


# Funciones de visualización
def visualize_preprocessing_steps(image_path):
    """
    Visualiza los pasos de preprocesamiento para una imagen
    """
    # Cargar imagen original
    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Aplicar pasos de preprocesamiento
    normalized = normalize_image(orig_img)
    cropped = crop_brain_region(normalized)
    clahe = apply_clahe(cropped)
    blurred = apply_gaussian_blur(clahe, kernel_size=(3, 3))
    resized = cv2.resize(blurred, (640, 640))

    # Visualizar todos los pasos
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(orig_img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(normalized, cmap="gray")
    axes[1].set_title("Normalizado")
    axes[2].imshow(cropped, cmap="gray")
    axes[2].set_title("Recortado")
    axes[3].imshow(clahe, cmap="gray")
    axes[3].set_title("CLAHE")
    axes[4].imshow(resized, cmap="gray")
    axes[4].set_title("Final (Resized)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_augmented(image, bboxes, class_labels):
    """
    Visualiza una imagen con sus bounding boxes
    """
    plt.imshow(image, cmap="gray")
    for bbox, label in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        # Use Rectangle from matplotlib.patches (imported at the top)
        plt.gca().add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
        )
    plt.show()


# Funciones para aumentación de datos
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CLAHE(p=0.3),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


def augment_image(image, bboxes, class_labels):
    """
    Aplica aumentación de datos a una imagen y sus bounding boxes
    """
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return transformed["image"], transformed["bboxes"], transformed["class_labels"]


def create_augmented_dataset(input_dir, output_dir, augmentations_per_image=3):
    """
    Crea un dataset aumentado a partir de las imágenes procesadas
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(
        os.path.join(input_dir, "*.png")
    )

    print(f"Generando aumentaciones para {len(image_files)} imágenes...")

    for img_path in tqdm(image_files):
        # Obtener nombre base y ruta de etiqueta
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(input_dir, f"{base_name}.txt")

        # Cargar imagen
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Convertir a RGB para albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Cargar etiquetas en formato YOLO
        bboxes = []
        class_labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(int(class_id))

        # Generar aumentaciones
        for i in range(augmentations_per_image):
            # Aplicar transformaciones
            aug_image, aug_bboxes, aug_labels = augment_image(
                image, bboxes, class_labels
            )

            # Guardar imagen aumentada
            aug_filename = f"{base_name}_aug_{i}.jpg"
            aug_image_path = os.path.join(output_dir, aug_filename)
            cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            # Guardar etiquetas aumentadas
            if aug_bboxes:
                aug_label_path = os.path.join(output_dir, f"{base_name}_aug_{i}.txt")
                with open(aug_label_path, "w") as f:
                    for bbox, label in zip(aug_bboxes, aug_labels):
                        f.write(f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    print(
        f"Aumentación de datos completada. Total de imágenes generadas: {len(image_files) * augmentations_per_image}"
    )
