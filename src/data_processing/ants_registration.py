# src/data_processing/ants_registration.py
import ants
from pathlib import Path

def register_masks_to_mri(mri_path: Path, template_path: Path, mask_paths: list[Path], output_dir: Path):
    """
    Registra un conjunto de máscaras de segmentación desde un espacio de plantilla (e.g., MNI152)
    al espacio de una resonancia magnética (MRI) de paciente específica.

    Realiza un registro rígido seguido de uno afín para obtener una transformación compuesta,
    la cual se aplica de forma inversa a cada máscara.

    Args:
        mri_path (Path): La ruta al archivo NIfTI de la MRI del paciente (imagen fija).
        template_path (Path): La ruta al archivo NIfTI de la plantilla (imagen en movimiento, e.g., MNI152_T1_1mm.nii.gz).
        mask_paths (list[Path]): Una lista de rutas a los archivos NIfTI de las máscaras
                                 que se desean registrar (en el espacio de la plantilla).
        output_dir (Path): El directorio donde se guardarán las máscaras registradas.
    """
    print(f"\nProcesando MRI: {mri_path.name}")

    # Cargar imágenes ANTs
    mri_ants = ants.image_read(str(mri_path))
    template_ants = ants.image_read(str(template_path))

    # Realizar registro rígida entre la MRI del paciente (fija) y la plantilla (movimiento)
    print("  Realizando registro rígido...")
    reg_rigid = ants.registration(
        fixed=mri_ants,
        moving=template_ants,
        type_of_transform='Rigid',
        verbose=False # Se puede cambiar a True para ver el progreso detallado
    )

    # Realizar registro afín, inicializado con la transformación rígida
    print("  Realizando registro afín...")
    reg_affine = ants.registration(
        fixed=mri_ants,
        moving=template_ants,
        type_of_transform='Affine',
        initial_transform=reg_rigid['fwdtransforms'], # Continúa desde la transformación rígida
        verbose=False # Se puede cambiar a True para ver el progreso detallado
    )

    # Aplicar la transformación compuesta a cada máscara
    for mask_path in mask_paths:
        mask_name = mask_path.name.replace(".nii.gz", "") # Obtiene el nombre del archivo sin extensión
        print(f"    Registrando máscara: {mask_name}")

        mask_ants = ants.image_read(str(mask_path))

        # Aplicar transformación inversa (de espacio de plantilla a espacio de MRI del paciente)
        # Se usa 'nearestNeighbor' para máscaras para preservar los valores discretos de las etiquetas.
        mask_registered = ants.apply_transforms(
            fixed=mri_ants,
            moving=mask_ants,
            transformlist=reg_affine['fwdtransforms'], # Aplica la transformación compuesta (rígida + afín)
            interpolator='nearestNeighbor'
        )

        # Construir la ruta de salida para la máscara registrada
        # Se reemplaza '_registered.nii.gz' del nombre de la MRI por el id del paciente
        patient_id = mri_path.name.replace('_registered.nii.gz', '')
        output_filename = f"{patient_id}_{mask_name}_registered.nii.gz"
        output_file_path = output_dir / output_filename

        # Guardar la máscara registrada
        ants.image_write(mask_registered, str(output_file_path))
        print(f"      Máscara registrada guardada en: {output_file_path}")


# Proceso principal

def main(registered_mri_dir: Path, template_masks_dir: Path, output_registered_masks_dir: Path):
    """
    Función principal que orquesta el proceso de registro de máscaras de segmentación.
    Define los directorios de entrada/salida, las máscaras a registrar y la plantilla,
    luego itera sobre las MRIs de los pacientes para aplicar el registro.
    """

    # Asegurarse de que el directorio de salida exista
    output_registered_masks_dir.mkdir(parents=True, exist_ok=True)

    # Lista de nombres de archivos de máscaras a registrar
    mask_filenames = [
        "left_hippocampus.nii.gz",
        "right_hippocampus.nii.gz",
        "left_putamen.nii.gz",
        "right_putamen.nii.gz"
    ]

    # Rutas completas de las máscaras en el espacio de la plantilla
    mask_paths = [template_masks_dir / f for f in mask_filenames]

    # Ruta de la plantilla MNI152
    template_path = template_masks_dir / "MNI152_T1_1mm.nii.gz"

    # Obtener todas las rutas de las MRIs de pacientes registradas
    mri_paths = [f for f in registered_mri_dir.glob("*_registered.nii.gz")]

    # Procesar cada MRI de paciente
    if not mri_paths:
        print(f"No se encontraron archivos MRI registrados en: {registered_mri_dir}. Asegúrate de que las MRIs están en el formato '_registered.nii.gz'.")
        return

    for mri_path in mri_paths:
        try:
            register_masks_to_mri(mri_path, template_path, mask_paths, output_registered_masks_dir)
        except Exception as e:
            print(f"Error procesando {mri_path.name}: {e}")

    print("\n✅ Proceso de registro de máscaras completado.")

if __name__ == "__main__":
    from src.config import DATA_DIR, TEMPLATE_MASKS_DIR, MASKS_DIR

    main(DATA_DIR, TEMPLATE_MASKS_DIR, MASKS_DIR)