# src/data_processing/data_loader.py
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from monai.transforms import (
    Compose, EnsureChannelFirst, LoadImage, Resize, ScaleIntensity, ToTensor
)
from src.config import REGIONS, LABEL_MAP, IMAGE_SIZE

IMAGE_TRANSFORMS = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize(spatial_size=IMAGE_SIZE),
    ToTensor()
])

class MultitaskSegmentationDataset(Dataset):
    """
    Dataset para la tarea de segmentación multi-región.

    Args:
        samples (list): Lista de diccionarios, donde cada diccionario contiene
                        las rutas de la imagen y las máscaras, y la etiqueta.
        transform (callable, optional): Transformaciones a aplicar a las imágenes
                                        y máscaras. Por defecto es None.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        """
        Devuelve el número total de muestras en el dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Obtiene una muestra del dataset en el índice especificado.

        Args:
            idx (int): Índice de la muestra a recuperar.

        Returns:
            tuple: Una tupla que contiene la imagen transformada y las máscaras
                   concatenadas en un solo tensor.
        """
        sample = self.samples[idx]
        image = self.transform(str(sample['image']))
        masks = []
        for region in REGIONS:
            mask = self.transform(str(sample['masks'][region]))
            masks.append(mask)
        masks = torch.cat(masks, dim=0)
        return image, masks

class FeatureExtractionDataset(Dataset):
    """
    Dataset para la extracción de características, utilizado para la clasificación.

    Args:
        samples (list): Lista de diccionarios, donde cada diccionario contiene
                        las rutas de la imagen y las etiquetas.
        transform (callable, optional): Transformaciones a aplicar a las imágenes.
                                        Por defecto es None.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        """
        Devuelve el número total de muestras en el dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Obtiene una muestra del dataset en el índice especificado.

        Args:
            idx (int): Índice de la muestra a recuperar.

        Returns:
            tuple: Una tupla que contiene la imagen transformada y su etiqueta.
        """
        sample = self.samples[idx]
        image = self.transform(str(sample['image']))
        label = sample['label']
        return image, label

def load_all_data(data_dir: Path, masks_dir: Path) -> list:
    """
    Carga los datos de las imágenes y sus máscaras asociadas, junto con las etiquetas.

    Args:
        data_dir (Path): Directorio donde se encuentran los datos de imagen.
        masks_dir (Path): Directorio donde se encuentran las máscaras de segmentación.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario representa una
              muestra con la ruta de la imagen, las rutas de las máscaras por
              región y la etiqueta de la condición.
    """
    with open(data_dir / "transform_summary.json", "r") as f:
        data_summary = json.load(f)

    full_data = []
    for pid, info in data_summary.items():
        image_path = data_dir / info["registered_nifti"]
        label = LABEL_MAP[info["condition"]]
        masks = {}
        all_masks_exist = True
        for region in REGIONS:
            mask_path = masks_dir / f"{pid}_{region}_registered.nii.gz"
            if not mask_path.exists():
                all_masks_exist = False
                break
            masks[region] = mask_path

        if all_masks_exist:
            full_data.append({"image": image_path, "masks": masks, "label": label})
    return full_data