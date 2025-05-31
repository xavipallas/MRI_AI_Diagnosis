# src/visualization/plot_mris.py
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
from pathlib import Path

def load_transform_summary(json_path: Path) -> dict:
    """
    Carga el resumen de transformaciones desde un archivo JSON.

    Args:
        json_path (Path): La ruta al archivo JSON que contiene el resumen de transformaciones.

    Returns:
        dict: Un diccionario con el resumen de transformaciones.

    Raises:
        FileNotFoundError: Si el archivo JSON no se encuentra.
        json.JSONDecodeError: Si hay un error al decodificar el archivo JSON.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"El archivo JSON no se encontró en: {json_path}")
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error al decodificar el archivo JSON: {e}", doc=f.read(), pos=0)

def group_patients_by_condition(transform_summary: dict) -> defaultdict:
    """
    Agrupa los pacientes por su condición (Alzheimer, Parkinson, Control)
    a partir del resumen de transformaciones.

    Args:
        transform_summary (dict): Diccionario con el resumen de transformaciones,
                                  donde cada paciente tiene una 'condition' y 'registered_nifti'.

    Returns:
        defaultdict: Un diccionario donde las claves son las condiciones y los valores son listas de tuplas,
                     cada tupla contiene (patient_id, registered_nifti_filename).
    """
    grouped = defaultdict(list)
    for patient_id, info in transform_summary.items():
        grouped[info["condition"]].append((patient_id, info["registered_nifti"]))
    return grouped

def select_balanced_samples(grouped_data: defaultdict, samples_per_group: int = 3) -> list:
    """
    Selecciona un número balanceado de muestras de cada grupo (condición).

    Args:
        grouped_data (defaultdict): Datos de pacientes agrupados por condición.
        samples_per_group (int): El número máximo de muestras a seleccionar por cada grupo.
                                 El valor real será el mínimo entre este y el número de pacientes disponibles
                                 en el grupo más pequeño.

    Returns:
        list: Una lista de tuplas, cada una conteniendo (patient_id, nifti_filename, condition)
              para las muestras seleccionadas.
    """
    # Determina el número mínimo de pacientes entre las condiciones para balancear la selección
    min_patients = float('inf')
    for condition in ["Alzheimer", "Parkinson", "Control"]:
        min_patients = min(min_patients, len(grouped_data[condition]))

    actual_samples_per_group = min(samples_per_group, min_patients)

    selected = []
    for condition in ["Alzheimer", "Parkinson", "Control"]:
        # Toma solo el número de muestras determinado para balancear
        selected.extend([(pid, nifti, condition) for pid, nifti in grouped_data[condition][:actual_samples_per_group]])
    return selected

def plot_nifti_slices(selected_samples: list, registered_dir: Path, cols: int = 3, figsize_base_height: int = 4):
    """
    Carga y muestra el corte axial central de las imágenes NIfTI seleccionadas.

    Args:
        selected_samples (list): Una lista de tuplas, cada una conteniendo
                                 (patient_id, nifti_filename, condition) de las muestras a mostrar.
        registered_dir (Path): La ruta al directorio donde se encuentran los archivos NIfTI registrados.
        cols (int): El número de columnas para organizar los subplots en la figura.
        figsize_base_height (int): La altura base de la figura por fila de subplots.
    """
    total_samples = len(selected_samples)
    rows = (total_samples + cols - 1) // cols # Calcula el número de filas necesarias

    plt.figure(figsize=(12, figsize_base_height * rows)) # Ajusta el tamaño de la figura

    for i, (patient_id, nifti_filename, condition) in enumerate(selected_samples):
        nifti_path = registered_dir / nifti_filename # Construye la ruta completa al archivo NIfTI

        try:
            # Cargar imagen NIfTI
            img = nib.load(nifti_path)
            data = img.get_fdata()

            # Obtiene el corte axial central
            slice_idx = data.shape[2] // 2
            slice_axial = data[:, :, slice_idx]

            # Muestra el corte en un subplot
            plt.subplot(rows, cols, i + 1)
            plt.imshow(np.rot90(slice_axial), cmap='gray') # Muestra la imagen, rotada para una mejor visualización
            plt.title(f"{patient_id} - {condition}", fontsize=10)
            plt.axis('off') # Desactiva los ejes
        except Exception as e:
            print(f"Error al cargar o procesar {nifti_filename} para el paciente {patient_id}: {e}")
            plt.subplot(rows, cols, i + 1)
            plt.text(0.5, 0.5, "Error al cargar", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title(f"{patient_id} - {condition}", fontsize=10)
            plt.axis('off')

    plt.tight_layout() # Ajusta automáticamente los parámetros de los subplots
    plt.show() # Muestra la figura con todas las imágenes


# Programa principal

def main(registered_dir: Path):
    """
    Función principal para la visualización de imágenes NIfTI registradas.
    Carga el resumen de transformaciones, agrupa los pacientes por condición,
    selecciona un número balanceado de muestras y las visualiza.
    """
    # Ruta de entrada
    json_path = registered_dir / "transform_summary.json"

    try:
        # Cargar el resumen de transformaciones
        transform_summary = load_transform_summary(json_path)

        # Agrupar pacientes por condición
        grouped_data = group_patients_by_condition(transform_summary)

        # Definir el número de muestras por grupo
        samples_to_show_per_group = 3
        
        # Seleccionar imágenes balanceadas de cada grupo
        selected_images = select_balanced_samples(grouped_data, samples_to_show_per_group)
        
        if not selected_images:
            print("No se encontraron imágenes para mostrar. Verifica la configuración de los datos o el número de muestras.")
            return

        # Mostrar las imágenes
        plot_nifti_slices(selected_images, registered_dir)

    except FileNotFoundError as e:
        print(f"Error: {e}. Asegúrate de que el directorio y el archivo JSON existen.")
    except json.JSONDecodeError as e:
        print(f"Error al leer el archivo JSON: {e}. Verifica que el archivo esté bien formado.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    from src.config import DATA_DIR
    main(DATA_DIR)