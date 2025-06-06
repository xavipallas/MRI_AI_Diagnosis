# src/config.py
from pathlib import Path
import torch

# Configuración Global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directorios de datos (rutas relativas a la raíz del proyecto)
RAW_DICOM_DIR = Path("data/raw_dicom_example") # MRIs en serie dicom
DATA_DIR = Path("data/registered_nifti_example") # MRIs registradas (output de dicom_to_nifti.py)
MASKS_DIR = Path("data/processed_segmentation_masks") # Máscaras segmentadas y registradas por ANTs (output de ants_registration.py)
# DIRECTORIO para máscaras de plantilla (entrada para ants_registration.py)
TEMPLATE_MASKS_DIR = Path("data/segmentation_masks_template")

# Regiones de interés para la segmentación.
REGIONS = [
    "left_hippocampus",
    "right_hippocampus",
    "left_putamen",
    "right_putamen"
]

# Mapping para el nombre completo de las regiones
REGION_FULL_NAMES = {
    "left_hippocampus": "Hipocampo Izquierdo",
    "right_hippocampus": "Hipocampo Derecho",
    "left_putamen": "Putamen Izquierdo",
    "right_putamen": "Putamen Derecho"
}

# Rutas de modelos pre-entrenados (relativas a la raíz del proyecto)
UNET_MODEL_PATH = Path("models/unet_multitask.pth")
XGB_CLASSIFIER_PATH = Path("models/xgboost_classifier.joblib")

# Otras configuraciones
IMAGE_SIZE = (96, 96, 96) # Tamaño para redimensionar las imágenes
UNET_CHANNELS = (16, 32, 64, 128, 256)
UNET_STRIDES = (2, 2, 2, 2)
UNET_NUM_RES_UNITS = 2

# Mapas de etiquetas para clasificación
LABEL_MAP = {"Alzheimer": 0, "Parkinson": 1, "Control": 2}

# Mapeo de etiquetas numéricas a nombres legibles
CLASS_LABELS = {0: "Alzheimer", 1: "Parkinson", 2: "Control"}