# src/segmentation/unet_inference.py
import torch
from monai.transforms import (
    Compose, EnsureChannelFirst, LoadImage, Resize, ScaleIntensity, ToTensor
)
from src.config import DEVICE, UNET_MODEL_PATH, IMAGE_SIZE, REGIONS
from src.segmentation.unet_architecture import UNet

INFERENCE_TRANSFORMS = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize(spatial_size=IMAGE_SIZE),
    ToTensor()
])

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
        model.eval() # Poner el modelo en modo evaluación
        print(f"Modelo UNet cargado exitosamente desde {UNET_MODEL_PATH}.")
    except FileNotFoundError:
        print(f"Error: Modelo UNet no encontrado en {UNET_MODEL_PATH}. Asegúrate de haberlo entrenado y guardado.")
        raise 
    except Exception as e:
        print(f"Error al cargar el modelo UNet: {e}")
        raise
    return model

def perform_unet_segmentation(model: UNet, image_path: str):
    """
    Realiza la segmentación de una imagen MRI utilizando el modelo UNet.

    Args:
        model (UNet): El modelo UNet pre-entrenado.
        image_path (str): El directorio de la imagen MRI.

    Returns:
        torch.Tensor: Las máscaras segmentadas binarizadas como un tensor de PyTorch
                      (num_regions, D, H, W) en CPU.
    """
    image = INFERENCE_TRANSFORMS(image_path)
    image = image.to(DEVICE)
    with torch.no_grad():
        output = model(image.unsqueeze(0)) # Añadir dimensión de batch
    binarized_masks = (torch.sigmoid(output) > 0.5).float().squeeze(0).cpu() # Eliminar dimensión de batch
    return binarized_masks