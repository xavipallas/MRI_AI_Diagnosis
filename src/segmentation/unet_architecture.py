# src/segmentation/unet_architecture.py
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from src.config import UNET_CHANNELS, UNET_STRIDES, UNET_NUM_RES_UNITS

def build_unet_multitask(num_regions: int, device: torch.device) -> UNet:
    """
    Construye y devuelve un modelo UNet para segmentación multiespectral.

    Args:
        num_regions (int): Número de canales de salida, igual al número de regiones a segmentar.
        device (torch.device): Dispositivo (CPU o GPU) donde se cargará el modelo.

    Returns:
        UNet: El modelo UNet configurado.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_regions,
        channels=UNET_CHANNELS,
        strides=UNET_STRIDES,
        num_res_units=UNET_NUM_RES_UNITS
    )
    return model.to(device)