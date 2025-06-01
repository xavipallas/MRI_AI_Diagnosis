# src/classification/feature_extraction.py
import numpy as np
import torch
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import DEVICE, REGIONS, REGION_FULL_NAMES
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
    Extrae características de forma (área 2D, excentricidad, solidez) de un corte axial central
    de cada máscara segmentada.

    Args:
        mask_tensor (torch.Tensor): Tensor de las máscaras segmentadas
                                     (num_regions, D, H, W).

    Returns:
        list: Una lista aplanada de características de forma para todas las regiones.
    """
    features = []
    for i in range(mask_tensor.shape[0]):
        # Asegurarse de que el tensor sea 3D para el slice y luego 2D para skimage
        img_np = (mask_tensor[i].squeeze().numpy() > 0).astype(np.uint8) # Eliminar posibles dims extra y binarizar
        
        # Se toma un corte axial central 2D para las características de forma.
        if img_np.ndim == 3: # Si es 3D, toma el slice central
            slice_2d = img_np[img_np.shape[0] // 2]
        elif img_np.ndim == 2: # Si ya es 2D (ej. si el input original era solo un slice)
            slice_2d = img_np
        else: # No es 2D ni 3D
            features.extend([0, 0, 0])
            continue

        labeled = label(slice_2d)
        props = regionprops(labeled)
        if props:
            vol_2d = props[0].area # "area" en 2D para la rebanada
            eccentricity = props[0].eccentricity
            solidity = props[0].solidity
            features.extend([vol_2d, eccentricity, solidity])
        else:
            features.extend([0, 0, 0]) # Si no se detecta ninguna región, añadir ceros
    return features

def extract_texture_features(mask_tensor: torch.Tensor) -> list:
    """
    Extrae características de textura (contraste, homogeneidad, energía, correlación)
    utilizando GLCM (Gray-Level Co-occurrence Matrix) de un corte axial central de cada máscara.

    Args:
        mask_tensor (torch.Tensor): Tensor de las máscaras segmentadas
                                     (num_regions, D, H, W).

    Returns:
        list: Una lista aplanada de características de textura para todas las regiones.
    """
    features = []
    for i in range(mask_tensor.shape[0]):
        # Convertir a imagen de 8 bits para GLCM, asegurando que haya datos
        img_np_original = mask_tensor[i].squeeze().numpy() # Eliminar posibles dims extra
        # Normalizar a 0-255 si tiene valores flotantes y luego convertir a uint8
        if img_np_original.max() > 0:
            img_np = (img_np_original / img_np_original.max() * 255).astype(np.uint8)
        else: # Si todos los valores son 0, la máscara está vacía
            img_np = np.zeros_like(img_np_original, dtype=np.uint8)

        # GLCM requiere una imagen 2D, se usa el corte central axial.
        if img_np.ndim == 3: # Si es 3D, toma el slice central
            slice_2d = img_np[img_np.shape[0] // 2]
        elif img_np.ndim == 2: # Si ya es 2D
            slice_2d = img_np
        else: # No es 2D ni 3D
            features.extend([0, 0, 0, 0])
            continue

        if np.sum(slice_2d) == 0 or slice_2d.max() == slice_2d.min():
            features.extend([0, 0, 0, 0]) # Si la región está vacía o es constante, no hay textura
            continue
        
        # Convertir a int para graycomatrix si no lo es
        slice_2d_int = slice_2d.astype(int)
        
        glcm = graycomatrix(slice_2d_int, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        features.extend([contrast, homogeneity, energy, correlation])
    return features

def get_feature_names() -> list[str]:
    """
    Genera los nombres de las características en el orden en que son extraídas.
    Necesario para interpretar la importancia de las características del XGBoost.

    Returns:
        list[str]: Una lista de cadenas con los nombres de las características.
    """
    feature_names = []
    for region in REGIONS:
        full_region_name = REGION_FULL_NAMES.get(region, region)
        feature_names.append(f"{full_region_name} - Volumen")
    for region in REGIONS:
        full_region_name = REGION_FULL_NAMES.get(region, region)
        feature_names.extend([
            f"{full_region_name} - Área 2D",
            f"{full_region_name} - Excentricidad",
            f"{full_region_name} - Solidez"
        ])
    for region in REGIONS:
        full_region_name = REGION_FULL_NAMES.get(region, region)
        feature_names.extend([
            f"{full_region_name} - Contraste",
            f"{full_region_name} - Homogeneidad",
            f"{full_region_name} - Energía",
            f"{full_region_name} - Correlación"
        ])
    return feature_names

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