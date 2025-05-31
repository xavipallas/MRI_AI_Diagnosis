# src/data_processing/dicom_to_nifti.py
import json
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm

class DICOMProcessingError(Exception):
    """Excepción personalizada para errores en el procesamiento de DICOM."""
    pass

def create_dicom_dataset(root_dir: Path) -> dict:
    """
    Crea un diccionario que organiza las rutas de los archivos DICOM por condición (Alzheimer, Parkinson, Control)
    y por paciente.

    Args:
        root_dir (Path): La ruta base donde se encuentran las carpetas de las condiciones y los pacientes.

    Returns:
        dict: Un diccionario anidado con la estructura:
              {'Condición': {'ID_Paciente': ['ruta/a/dicom1.dcm', 'ruta/a/dicom2.dcm', ...]}}
    """
    patient_data = {"Alzheimer": {}, "Parkinson": {}, "Control": {}}

    for condition_dir in root_dir.iterdir():
        if not condition_dir.is_dir():
            continue

        condition_name = condition_dir.name.lower()
        target_condition = None

        if "alzheimer" in condition_name:
            target_condition = "Alzheimer"
        elif "parkinson" in condition_name:
            target_condition = "Parkinson"
        elif "control" in condition_name:
            target_condition = "Control"
        else:
            continue  # Continúa con el siguiente directorio

        for patient_dir in condition_dir.iterdir():
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name
            image_paths = [str(img_file) for img_file in patient_dir.iterdir() if img_file.is_file()]

            if image_paths:
                patient_data[target_condition].setdefault(patient_id, []).extend(image_paths)
    return patient_data

## --- Utilidades de Procesamiento de Imágenes DICOM ---

def dicom_series_to_volume(slice_paths: list[str]) -> sitk.Image:
    """
    Convierte una serie de rutas de archivos DICOM en un volumen 3D de SimpleITK.
    Los cortes se ordenan basándose en la coordenada Z (SliceLocation) de los metadatos DICOM.

    Args:
        slice_paths (list[str]): Una lista de rutas a los archivos DICOM que forman una serie.

    Returns:
        sitk.Image: Un objeto SimpleITK.Image que representa el volumen 3D.

    Raises:
        DICOMProcessingError: Si la lista de rutas está vacía o si ocurre un error al leer los metadatos DICOM.
    """
    if not slice_paths:
        raise DICOMProcessingError("La lista de rutas de DICOM está vacía.")

    try:
        # Ordena las imágenes basándose en el metadato '0020|0032' (Image Position (Patient) Z-component)
        slice_paths_sorted = sorted(
            slice_paths,
            key=lambda p: float(sitk.ReadImage(p).GetMetaData("0020|0032").split("\\")[2])
        )
    except Exception as e:
        raise DICOMProcessingError(f"Error al ordenar las imágenes DICOM: {e}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(slice_paths_sorted)
    reader.MetaDataDictionaryArrayUpdateOn() # Asegura que los metadatos se carguen

    img = reader.Execute()
    return sitk.Cast(img, sitk.sitkFloat32) # Convierte la imagen a Float32 para consistencia en el procesamiento

def resample_isotropic(img: sitk.Image, iso: float = 1.0) -> sitk.Image:
    """
    Remuestrea una imagen de SimpleITK a un espaciado isotrópico.

    Args:
        img (sitk.Image): La imagen de SimpleITK de entrada.
        iso (float): El valor del espaciado isotrópico deseado (por defecto, 1.0 mm).

    Returns:
        sitk.Image: La imagen remuestreada con espaciado isotrópico.
    """
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # Calcula el nuevo tamaño de la imagen para el espaciado isotrópico
    new_size = [int(round(sz * sp / iso)) for sz, sp in zip(original_size, original_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear) # Usa interpolación lineal
    resampler.SetOutputSpacing([iso] * 3)     # Establece el espaciado de salida isotrópico
    resampler.SetSize(new_size)               # Establece el nuevo tamaño
    resampler.SetOutputDirection(img.GetDirection()) # Mantiene la dirección de la imagen original
    resampler.SetOutputOrigin(img.GetOrigin())     # Mantiene el origen de la imagen original

    return resampler.Execute(img)

def rigid_registration(fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.Transform:
    """
    Realiza un registro rígido (traslación y rotación) entre dos imágenes 3D
    utilizando el algoritmo de descenso de gradiente de paso regular y la información mutua de Mattes.

    Args:
        fixed_image (sitk.Image): La imagen fija a la que se alineará la imagen en movimiento.
        moving_image (sitk.Image): La imagen en movimiento que se transformará.

    Returns:
        sitk.Transform: La transformación rígida calculada que mapea la imagen en movimiento a la imagen fija.
    """
    # Asegura que ambas imágenes sean Float32 para el registro
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Inicializa la transformación en el centro de las imágenes
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(), # Transformación rígida 3D (Euler: rotación y traslación)
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) # Métrica de similitud
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)       # Estrategia de muestreo
    registration_method.SetMetricSamplingPercentage(0.2)                            # Porcentaje de muestreo
    registration_method.SetInterpolator(sitk.sitkLinear)                            # Interpolador

    # Configuración del optimizador (descenso de gradiente de paso regular)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8
    )
    registration_method.SetOptimizerScalesFromPhysicalShift() # Escala los parámetros del optimizador
    registration_method.SetInitialTransform(initial_transform, inPlace=False) # Establece la transformación inicial

    # Ejecuta el registro
    final_transform = registration_method.Execute(fixed_image, moving_image)
    return final_transform

def resample_to_ref(moving: sitk.Image, ref: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    """
    Remuestrea una imagen en movimiento a la misma rejilla (referencia) que una imagen fija,
    aplicando una transformación dada.

    Args:
        moving (sitk.Image): La imagen en movimiento a remuestrear.
        ref (sitk.Image): La imagen de referencia (fija) cuya geometría se utilizará.
        transform (sitk.Transform): La transformación a aplicar a la imagen en movimiento.

    Returns:
        sitk.Image: La imagen en movimiento remuestreada y transformada.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)      # Utiliza la imagen de referencia para la geometría de salida
    resampler.SetInterpolator(sitk.sitkLinear) # Usa interpolación lineal
    resampler.SetTransform(transform)     # Aplica la transformación

    return resampler.Execute(moving)

## Función Principal

def main(input_dicom_root_dir: Path, output_nifti_dir: Path):
    """
    Función principal para el procesamiento y registro de imágenes DICOM.
    Carga los datos de pacientes, selecciona una imagen de referencia,
    registra rígidamente todas las demás imágenes a la referencia
    y guarda los resultados (imágenes NIfTI registradas y transformaciones).
    """
    # Crea el directorio de salida si no existe
    output_nifti_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar y organizar los datos DICOM
    patient_data = create_dicom_dataset(input_dicom_root_dir)
    print("Pacientes cargados:")
    for cond, pats in patient_data.items():
        print(f"  {cond}: {len(pats)} pacientes")

    # 2. Seleccionar una imagen de referencia para el registro
    # Se selecciona el primer paciente de la condición "Control" como referencia.
    ref_cond = "Control"
    try:
        ref_id = next(iter(patient_data[ref_cond]))
    except StopIteration:
        print(f"No se encontraron pacientes en la condición '{ref_cond}'. No se puede seleccionar una referencia.")
        return

    ref_paths = patient_data[ref_cond][ref_id]
    print(f"\nUsando {ref_id} ({ref_cond}) como volumen fijo...")

    # Convierte la serie DICOM de referencia a volumen y la remuestrea isotrópicamente
    fixed_image = resample_isotropic(dicom_series_to_volume(ref_paths))
    sitk.WriteImage(fixed_image, output_nifti_dir / f"{ref_id}_fixed.nii.gz")
    print(f"Volumen fijo guardado como: {output_nifti_dir / f'{ref_id}_fixed.nii.gz'}")

    # 3. Registrar todas las demás imágenes a la referencia
    transform_summary = {} # Diccionario para almacenar el resumen de las transformaciones

    for cond, pats in patient_data.items():
        for pid, paths in tqdm(pats.items(), desc=f"Registrando {cond}", unit="paciente"):
            # Si es la imagen de referencia, la salta ya que ya está procesada
            if pid == ref_id and cond == ref_cond:
                continue

            try:
                # Convierte la serie DICOM del paciente a volumen y la remuestrea isotrópicamente
                moving_image = resample_isotropic(dicom_series_to_volume(paths))

                # Realiza el registro rígido
                transform = rigid_registration(fixed_image, moving_image)

                # Remuestrea la imagen en movimiento a la rejilla de la imagen fija con la transformación
                registered_image = resample_to_ref(moving_image, fixed_image, transform)

                # Guarda la imagen NIfTI registrada y la transformación
                registered_nifti_filename = f"{pid}_registered.nii.gz"
                transform_filename = f"{pid}.tfm"

                sitk.WriteImage(registered_image, output_nifti_dir / registered_nifti_filename)
                sitk.WriteTransform(transform, str(output_nifti_dir / transform_filename))

                # Almacena la información en el resumen
                transform_summary[pid] = {
                    "condition": cond,
                    "transform_file": transform_filename,
                    "registered_nifti": registered_nifti_filename
                }
            except DICOMProcessingError as e:
                print(f"Error procesando paciente {pid} en condición {cond}: {e}")
                continue # Continúa con el siguiente paciente
            except Exception as e:
                print(f"Error inesperado al procesar paciente {pid} en condición {cond}: {e}")
                continue # Continúa con el siguiente paciente


    # 4. Guardar el resumen de las transformaciones en un archivo JSON
    summary_filepath = output_nifti_dir / "transform_summary.json"
    with open(summary_filepath, "w") as f:
        json.dump(transform_summary, f, indent=2)
    print(f"\n✅ Registro completo. Resultados y resumen guardados en: {output_nifti_dir}")

if __name__ == "__main__":
    from src.config import RAW_DICOM_DIR, DATA_DIR

    main(RAW_DICOM_DIR, DATA_DIR)