# src/visualization/plot_mris_with_masks.py
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def load_nifti_data(nifti_path: Path) -> np.ndarray:
    """
    Carga los datos de una imagen NIfTI y devuelve un array NumPy.

    Args:
        nifti_path (Path): La ruta al archivo NIfTI.

    Returns:
        np.ndarray: Los datos de la imagen NIfTI como un array NumPy.

    Raises:
        FileNotFoundError: Si el archivo NIfTI no existe.
        nibabel.filebased.FileBasedError: Si hay un error al cargar el archivo NIfTI.
    """
    if not nifti_path.exists():
        raise FileNotFoundError(f"El archivo NIfTI no se encontró en: {nifti_path}")
    
    try:
        img = nib.load(nifti_path)
        return img.get_fdata()
    except nib.filebased.FileBasedError as e:
        raise nibabel.filebased.FileBasedError(f"Error al cargar el archivo NIfTI {nifti_path}: {e}")

def prepare_mri_for_display(mri_data: np.ndarray) -> np.ndarray:
    """
    Prepara el corte axial central de los datos de MRI para su visualización.
    Normaliza los valores de intensidad y convierte la imagen a un formato RGB.

    Args:
        mri_data (np.ndarray): Los datos 3D de la MRI.

    Returns:
        np.ndarray: Una imagen 2D de la MRI normalizada en formato RGB (altura, ancho, 3).
    """
    # Obtiene el corte axial central
    slice_idx = mri_data.shape[2] // 2
    mri_slice = mri_data[:, :, slice_idx]

    # Normaliza el corte de MRI a un rango de [0, 1]
    min_val, max_val = mri_slice.min(), mri_slice.max()
    if max_val - min_val > 0: # Evita división por cero si la imagen es constante
        mri_slice = (mri_slice - min_val) / (max_val - min_val)
    else:
        mri_slice = np.zeros_like(mri_slice) # Si es constante, llena de ceros

    # Convierte el corte de un solo canal a una imagen RGB replicando el canal
    return np.stack([mri_slice] * 3, axis=-1)

def overlay_masks_on_mri(
    mri_rgb_slice: np.ndarray, 
    patient_id: str, 
    mask_dir: Path, 
    mask_colors: dict
) -> np.ndarray:
    """
    Superpone máscaras de segmentación sobre un corte MRI RGB.

    Para cada máscara especificada, carga el corte central, lo binariza,
    y mezcla sus píxeles con el color definido en las regiones correspondientes
    del corte MRI RGB.

    Args:
        mri_rgb_slice (np.ndarray): El corte MRI en formato RGB (altura, ancho, 3).
        patient_id (str): El ID del paciente actual.
        mask_dir (Path): El directorio donde se encuentran las máscaras registradas.
        mask_colors (dict): Un diccionario que mapea nombres de regiones a colores RGB (tuplas de 3 floats).

    Returns:
        np.ndarray: La imagen MRI RGB con las máscaras superpuestas.
    """
    overlapped_mri_rgb = np.copy(mri_rgb_slice) # Trabaja sobre una copia para no modificar el original

    for region, color in mask_colors.items():
        mask_path = mask_dir / f"{patient_id}_{region}_registered.nii.gz"

        if not mask_path.exists():
            print(f"⚠️  No se encontró la máscara: {mask_path}. Saltando esta máscara.")
            continue

        try:
            mask_data = load_nifti_data(mask_path)
            # Obtiene el corte central y lo binariza (valores > 0 son parte de la máscara)
            mask_slice = mask_data[:, :, mask_data.shape[2] // 2] > 0

            # Superpone la máscara en la imagen RGB.
            # Se usa una mezcla (0.5 * MRI_pixel + 0.5 * mask_color) para mantener la visibilidad de la MRI
            for i in range(3):  # Iterar sobre canales R, G, B
                overlapped_mri_rgb[:, :, i][mask_slice] = (
                    0.5 * overlapped_mri_rgb[:, :, i][mask_slice] + 0.5 * color[i]
                )
        except Exception as e:
            print(f"Error al superponer la máscara {mask_path.name}: {e}")
            continue
            
    return overlapped_mri_rgb

def visualize_mri_with_masks(
    mri_rgb_with_masks: np.ndarray, 
    patient_id: str, 
    figsize: tuple = (6, 6)
):
    """
    Muestra un corte MRI con máscaras de segmentación superpuestas.

    Args:
        mri_rgb_with_masks (np.ndarray): El corte MRI con máscaras superpuestas en formato RGB.
        patient_id (str): El ID del paciente para el título de la figura.
        figsize (tuple): El tamaño de la figura para la visualización.
    """
    plt.figure(figsize=figsize)
    plt.imshow(np.rot90(mri_rgb_with_masks)) # Rota la imagen para una orientación correcta
    plt.title(f'Paciente {patient_id} - Máscaras superpuestas')
    plt.axis('off') # Desactiva los ejes para una visualización más limpia
    plt.show()

## Proceso principal

def main(mri_dir: Path, mask_dir: Path):
    """
    Función principal para la visualización de MRIs con máscaras de segmentación superpuestas.
    Define las rutas, los colores de las máscaras, carga y procesa las imágenes
    y muestra los resultados para un número limitado de pacientes.
    """
    # Define los colores para cada región de la máscara en formato RGB (valores de 0 a 1)
    mask_colors = {
        'left_hippocampus': (1.0, 0.0, 0.0),    # Rojo
        'right_hippocampus': (0.0, 1.0, 0.0),   # Verde
        'left_putamen': (0.0, 0.0, 1.0),        # Azul
        'right_putamen': (1.0, 1.0, 0.0),       # Amarillo
    }

    # Obtiene las rutas de los archivos MRI registrados, limitando a los primeros 5 pacientes para visualización
    # Asegúrate de que el patrón de nombre de archivo sea correcto para tus MRIs registradas
    mri_files_list = sorted(list(mri_dir.glob('*_registered.nii.gz')))[:5]

    if not mri_files_list:
        print(f"No se encontraron archivos MRI registrados en: {mri_dir}. Verifica la ruta y el patrón de nombre de archivo.")
        return

    # Itera sobre cada archivo MRI seleccionado
    for mri_path in mri_files_list:
        # Extrae el ID del paciente del nombre del archivo MRI (asumiendo formato 'ID_registered.nii.gz')
        patient_id = mri_path.name.split('_')[0]

        try:
            # Carga los datos de la MRI
            mri_data = load_nifti_data(mri_path)
            
            # Prepara el corte MRI para mostrar
            mri_rgb = prepare_mri_for_display(mri_data)

            # Superpone las máscaras en el corte MRI
            mri_with_overlays = overlay_masks_on_mri(mri_rgb, patient_id, mask_dir, mask_colors)

            # Muestra la imagen resultante
            visualize_mri_with_masks(mri_with_overlays, patient_id)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Ocurrió un error inesperado al procesar al paciente {patient_id}: {e}")

if __name__ == "__main__":
    from src.config import DATA_DIR, MASKS_DIR
    main(DATA_DIR, MASKS_DIR)