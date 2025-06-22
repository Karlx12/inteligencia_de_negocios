"""
Script simplificado para subir todas las imágenes a Roboflow en un solo batch
"""

import os
import requests
import base64
import time
import glob
from tqdm import tqdm
from pathlib import Path


class SimpleRoboflowUploader:
    def __init__(self, api_key, workspace_id, project_id):
        """Inicializa el uploader"""
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.project_id = project_id
        self.base_url = "https://api.roboflow.com"
        self.uploaded_count = 0
        self.failed_count = 0

    def upload_image(self, image_path, batch_name="annotation_batch"):
        """Sube una imagen individual a Roboflow"""
        try:
            # Leer y codificar la imagen
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                encoded_image = base64.b64encode(image_data).decode("utf-8")

            # URL de la API
            url = (
                f"{self.base_url}/dataset/{self.workspace_id}/{self.project_id}/upload"
            )

            # Payload
            payload = {
                "api_key": self.api_key,
                "name": os.path.basename(image_path),
                "image": encoded_image,
                "split": "train",
                "batch": batch_name,
            }

            # Realizar petición
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                self.uploaded_count += 1
                return True
            else:
                self.failed_count += 1
                print(f"Error {response.status_code}: {os.path.basename(image_path)}")
                return False

        except Exception as e:
            self.failed_count += 1
            print(f"Excepción: {os.path.basename(image_path)} - {str(e)}")
            return False

    def upload_all_images(
        self, dataset_path, batch_name="brain_tumor_batch", delay=0.1
    ):
        """Sube todas las imágenes del dataset a un solo batch"""

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"❌ Error: No existe la ruta {dataset_path}")
            return

        # Encontrar todas las imágenes
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        all_images = []

        print("🔍 Buscando imágenes...")
        for ext in image_extensions:
            # Buscar recursivamente en todas las subcarpetas
            pattern = dataset_path / "**" / ext
            found_images = list(dataset_path.glob(f"**/{ext}"))
            found_images.extend(list(dataset_path.glob(f"**/{ext.upper()}")))
            all_images.extend(found_images)

        # Eliminar duplicados
        all_images = list(set(all_images))

        print(f"📊 Total de imágenes encontradas: {len(all_images)}")
        print(f"📦 Batch de destino: {batch_name}")

        if not all_images:
            print("❌ No se encontraron imágenes")
            return

        # Confirmar antes de proceder
        response = input(f"\n¿Subir {len(all_images)} imágenes a Roboflow? (y/N): ")
        if response.lower() not in ["y", "yes", "sí", "s"]:
            print("❌ Cancelado")
            return

        print(f"\n🚀 Iniciando subida...")

        # Subir con barra de progreso
        for image_path in tqdm(all_images, desc="Subiendo imágenes"):
            self.upload_image(str(image_path), batch_name)

            # Delay para evitar rate limiting
            if delay > 0:
                time.sleep(delay)

        # Resumen
        self.print_summary()

    def print_summary(self):
        """Imprime resumen final"""
        print(f"\n{'=' * 50}")
        print(f"📊 RESUMEN FINAL")
        print(f"{'=' * 50}")
        print(f"✅ Imágenes subidas: {self.uploaded_count}")
        print(f"❌ Fallos: {self.failed_count}")
        print(
            f"📈 Tasa de éxito: {(self.uploaded_count / (self.uploaded_count + self.failed_count) * 100):.1f}%"
        )
        print(f"{'=' * 50}")


def main():
    """Función principal"""

    # Configuración desde config.py
    try:
        import config

        api_key = config.API_KEY
        workspace_id = config.WORKSPACE_ID
        project_id = config.PROJECT_ID
        dataset_path = config.DATASET_PATH
    except ImportError:
        print("❌ Error: No se encontró config.py")
        return
    except AttributeError as e:
        print(f"❌ Error en config.py: {e}")
        return

    # Validaciones
    if api_key == "YOUR_ROBOFLOW_API_KEY":
        print("❌ Error: Configura tu API_KEY en config.py")
        return

    print("🎯 UPLOAD SIMPLIFICADO A ROBOFLOW")
    print(f"📁 Dataset: {dataset_path}")
    print(f"🏢 Workspace: {workspace_id}")
    print(f"📋 Proyecto: {project_id}")

    # Crear uploader y ejecutar
    uploader = SimpleRoboflowUploader(api_key, workspace_id, project_id)
    uploader.upload_all_images(
        dataset_path=dataset_path, batch_name="brain_tumor_annotations", delay=0.1
    )

    print("✅ Proceso completado!")


if __name__ == "__main__":
    main()
