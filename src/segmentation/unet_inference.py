# src/segmentation/unet_inference.py
import torch
import numpy as np
from monai.transforms import Resize
from src.segmentation.unet_architecture import build_unet_multitask
from src.config import DEVICE, UNET_MODEL_PATH, IMAGE_SIZE, REGIONS
from src.segmentation.unet_architecture import UNet


def preprocess_mri(nifti_data: np.ndarray) -> torch.Tensor:
    """
    Aplica las transformaciones definidas a los datos NIfTI de una MRI.
    Asegura que los datos estén en el formato correcto para el modelo UNet.

    Args:
        nifti_data (np.ndarray): Datos de la imagen MRI cargados de un archivo NIfTI.

    Returns:
        torch.Tensor: El tensor de la imagen MRI preprocesada, listo para la entrada al modelo.
    """
    # EnsureChannelFirst: añade la dimensión de canal (1, D, H, W)
    # MONAI espera (C, D, H, W) para imágenes 3D
    data_with_channel = np.expand_dims(nifti_data, axis=0) 

    # ScaleIntensity: normaliza al rango [0, 1] o similar
    min_val, max_val = data_with_channel.min(), data_with_channel.max()
    if max_val - min_val > 0:
        scaled_data = (data_with_channel - min_val) / (max_val - min_val)
    else:
        # Manejar el caso de una imagen constante para evitar división por cero
        scaled_data = np.zeros_like(data_with_channel) 

    # Convertir a tensor de PyTorch antes de Resize
    input_tensor = torch.from_numpy(scaled_data).float()

    # Resize: a (96, 96, 96)
    # Asegurarse de que esta transformación se aplica a un tensor.
    resized_tensor = Resize(spatial_size=IMAGE_SIZE)(input_tensor)

    # ToTensor: ya es un tensor, por lo que este paso es esencialmente una no-operación aquí
    return resized_tensor


def load_unet_model() -> UNet:
    """
    Carga un modelo UNet pre-entrenado.

    Returns:
        UNet: El modelo UNet cargado y configurado en modo de evaluación.

    Raises:
        FileNotFoundError: Si el archivo del modelo no se encuentra.
        Exception: Para otros errores durante la carga del modelo.
    """
    model = build_unet_multitask(len(REGIONS), DEVICE)
    
    try:
        model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Modelo UNet cargado exitosamente desde {UNET_MODEL_PATH}.")
    except FileNotFoundError:
        print(f"Error: Modelo UNet no encontrado en {UNET_MODEL_PATH}. Asegúrate de haberlo entrenado y guardado.")
        raise 
    except Exception as e:
        print(f"Error al cargar el modelo UNet: {e}")
        raise
    return model


def perform_segmentation(model: UNet, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Realiza la segmentación de una imagen MRI utilizando el modelo UNet.

    Args:
        model (UNet): El modelo UNet pre-entrenado.
        image_tensor (torch.Tensor): El tensor de la imagen MRI preprocesada.

    Returns:
        torch.Tensor: Las máscaras segmentadas binarizadas como un tensor de PyTorch
                      (num_regions, D, H, W) en CPU.
    """
    with torch.no_grad():
        # Añadir dimensión de batch (B, C, D, H, W) y mover a dispositivo
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        outputs = model(image_tensor)
        # Aplicar sigmoide y binarizar las predicciones
        # Quitar dimensión de batch, mover a CPU
        binarized_masks = (torch.sigmoid(outputs) > 0.5).float().squeeze(0).cpu()
    return binarized_masks