# src/classification/feature_extraction.py
import numpy as np
import torch
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import DEVICE
from src.segmentation.unet_architecture import UNet
from src.data_processing.data_loader import FeatureExtractionDataset

def extract_region_volumes(mask_tensor: torch.Tensor) -> list:
    """
    Calcula el volumen (número de píxeles/voxels) de cada región en un tensor de máscara.

    Args:
        mask_tensor (torch.Tensor): Tensor de máscaras binarizadas.

    Returns:
        list: Una lista de volúmenes para cada región.
    """
    return [m.sum().item() for m in mask_tensor]

def extract_shape_features(mask_tensor: torch.Tensor) -> list:
    """
    Extrae características de forma 2D (área, excentricidad, solidez) de cada región
    a partir de un corte axial central de la máscara.

    Args:
        mask_tensor (torch.Tensor): Tensor de máscaras binarizadas.

    Returns:
        list: Una lista concatenada de características de forma para todas las regiones.
              Si una región no tiene propiedades, se añaden ceros.
    """
    features = []
    for i in range(mask_tensor.shape[0]):
        img_np = (mask_tensor[i].numpy() > 0).astype(np.uint8)
        # Se toma un corte axial central 2D para las características de forma.
        slice_2d = img_np[img_np.shape[0] // 2]
        labeled = label(slice_2d)
        props = regionprops(labeled)
        if props:
            # area en 2D
            vol_2d = props[0].area
            eccentricity = props[0].eccentricity
            solidity = props[0].solidity
            features.extend([vol_2d, eccentricity, solidity])
        else:
            features.extend([0, 0, 0])  # Si no se detectan propiedades, se añaden ceros
    return features

def extract_texture_features(mask_tensor: torch.Tensor) -> list:
    """
    Extrae características de textura (contraste, homogeneidad, energía, correlación)
    utilizando GLCM (Gray-Level Co-occurrence Matrix) de cada región a partir de
    un corte axial central de la máscara.

    Args:
        mask_tensor (torch.Tensor): Tensor de máscaras binarizadas.

    Returns:
        list: Una lista concatenada de características de textura para todas las regiones.
    """
    features = []
    for i in range(mask_tensor.shape[0]):
        img_np = (mask_tensor[i].numpy() * 255).astype(np.uint8)
        # GLCM requiere una imagen 2D, se usa el corte central axial.
        slice_2d = img_np[img_np.shape[0] // 2]
        if np.sum(slice_2d) == 0:  # Evitar error si la región está vacía
            features.extend([0, 0, 0, 0])
            continue
        # Asegurarse de que slice_2d no esté vacío antes de calcular GLCM
        if slice_2d.max() == slice_2d.min(): # Para evitar error en graycomatrix si todos los valores son iguales
             glcm = np.zeros((256, 256, 1, 1))
        else:
            glcm = graycomatrix(slice_2d, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        features.extend([contrast, homogeneity, energy, correlation])
    return features

def extract_features(model: UNet, dataset: FeatureExtractionDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae características de volumen, forma y textura de las regiones segmentadas
    utilizando el modelo UNet entrenado.

    Args:
        model (UNet): El modelo UNet entrenado para la segmentación.
        dataset (FeatureExtractionDataset): Dataset con las imágenes y etiquetas.

    Returns:
        tuple[np.ndarray, np.ndarray]: Un array NumPy de características extraídas y
                                       un array NumPy de etiquetas correspondientes.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features, labels = [], []
    with torch.no_grad():
        for image, label in tqdm(loader, desc="Extrayendo características"):
            image = image.to(DEVICE)
            pred = torch.sigmoid(model(image))
            binarized = (pred > 0.5).float().cpu()  # Binarizar las predicciones y mover a CPU

            # Extraer características para cada muestra
            vols = extract_region_volumes(binarized[0])
            shape_feats = extract_shape_features(binarized[0])
            texture_feats = extract_texture_features(binarized[0])

            combined_features = vols + shape_feats + texture_feats
            features.append(combined_features)
            labels.append(label.item())
    return np.array(features), np.array(labels)