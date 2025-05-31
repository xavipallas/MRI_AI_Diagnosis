# MRI_AI_Diagnosis: Sistema de soporte al diagnóstico de trastornos neurológicos mediante MRI

![Diagnóstico IA MRI](https://img.shields.io/badge/Estado-Finalizado-green)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![Licencia](https://img.shields.io/github/license/xavipallas/MRI_AI_Diagnosis)
![Dependencias](https://img.shields.io/badge/Dependencias-Pipfile-green)

---

## 🚀 Visión general del proyecto

`MRI_AI_Diagnosis` es un sistema integral diseñado para asistir en el diagnóstico de trastornos neurológicos (Alzheimer y Parkinson) a partir de imágenes de Resonancia Magnética (MRI). El proyecto utiliza una combinación de **segmentación de imágenes médicas con redes neuronales UNet** y **clasificación basada en características con XGBoost**, todo orquestado a través de una **interfaz gráfica de usuario (GUI) interactiva con Streamlit**.

### Características principales:

* **Preprocesamiento de MRI:** Conversión de DICOM a NIfTI, registro y normalización de imágenes cerebrales.
* **Segmentación multitarea:** Un modelo UNet 3D entrenado para segmentar simultáneamente regiones clave del cerebro (hipocampos y putámenes) en las MRIs.
* **Extracción de características cuantitativas:** Cálculo automático de volúmenes, características de forma y textura a partir de las regiones segmentadas.
* **Clasificación avanzada:** Un clasificador XGBoost entrenado con las características extraídas para predecir la condición del paciente (Alzheimer, Parkinson, Control).
* **Interfaz de usuario intuitiva:** Una aplicación Streamlit que permite cargar imágenes MRI, realizar la segmentación y clasificación, y visualizar los resultados de manera interactiva.

---

## 🛠️ Estructura del proyecto

El repositorio está organizado de forma modular para facilitar la navegación, el desarrollo y el mantenimiento:

```
MRI_AI_Diagnosis/
├── .github/                      # Configuraciones de GitHub (ej. workflows de CI/CD)
├── data/                         # Almacena datos de ejemplo y plantillas
│   ├── raw_dicom_example/        # Ejemplos de MRIs en series DICOM
│   ├── registered_nifti_example/ # Ejemplos de MRIs registradas (salida del preprocesamiento)
│   ├── segmentation_masks_template/ # Máscaras de plantilla MNI y la plantilla de referencia
│   ├── processed_segmentation_masks/ # Máscaras de segmentación de MRIs
│   └── transform_summary.json    # Relación de transformación y condición de las MRIs registradas
├── models/                       # Modelos entrenados
│   ├── unet_multitask.pth        # Modelo UNet de segmentación
│   └── xgboost_classifier.joblib # Clasificador XGBoost
├── notebooks/                    # Cuadernos Jupyter para exploración, tutoriales y experimentación
├── src/                          # Código fuente modular del proyecto
│   ├── config.py                 # Configuraciones globales y constantes
│   ├── data_processing/          # Scripts para preprocesamiento y carga de datos
│   │   ├── dicom_to_nifti.py     # Conversión de DICOM a NIfTI y registro rígido
│   │   ├── ants_registration.py  # Registro de máscaras ANTs
│   │   └── data_loader.py        # Clases Dataset y funciones de carga de datos
│   ├── segmentation/             # Módulos relacionados con la segmentación UNet
│   │   ├── unet_architecture.py  # Definición de la arquitectura UNet
│   │   ├── unet_trainer.py       # Lógica de entrenamiento del UNet
│   │   └── unet_inference.py     # Lógica para la inferencia con el UNet
│   ├── classification/           # Módulos para extracción de características y clasificación
│   │   ├── feature_extraction.py # Funciones para extraer características de volumen, forma y textura
│   │   └── classifier_model.py   # Entrenamiento y carga del clasificador XGBoost
│   ├── visualization/            # Scripts para visualización de MRIs y máscaras
│   │   ├── plot_mris.py
│   │   └── plot_mris_with_masks.py
│   └── gui/                      # Módulo para la interfaz de usuario Streamlit
│       └── app.py                # Aplicación Streamlit principal
├── .gitignore                    # Archivos y directorios ignorados por Git
├── LICENSE                       # Información de licencia
├── README.md                     # Este archivo
└── requirements.txt              # Dependencias del proyecto
```

---

## ⚙️ Instalación

Sigue estos pasos para configurar el entorno y ejecutar el proyecto localmente.

### 1. Clona el repositorio

```
git clone [https://github.com/xavipallas/MRI_AI_Diagnosis.git](https://github.com/xavipallas/MRI_AI_Diagnosis.git)
cd MRI_AI_Diagnosis
```

### 2. Crea y activa el entorno virtual

```
python -m venv venv
```

* **En Windows:**

  ```
  .\venv\Scripts\activate
  ```
  (Si tienes problemas con la política de ejecución de PowerShell, puedes necesitar ejecutar Set-ExecutionPolicy          RemoteSigned como administrador una vez).
  
* **En macOS/Linux:**
  ```
  source venv/bin/activate
  ```

### 3. Instala las dependencias
Con tu entorno virtual activado, instala todas las librerías necesarias:
```
pip install -r requirements.txt
```

---

## 🚀 Uso
El proyecto puede usarse de varias maneras: a través de su GUI interactiva, o ejecutando scripts individuales para cada etapa del pipeline.

### 1. Ejecutar la Interfaz Gráfica de Usuario (GUI)
La forma más sencilla de interactuar con el sistema es a través de la aplicación Streamlit.

```
streamlit run src/gui/app.py
```
Esto abrirá la aplicación en tu navegador web, donde podrás cargar tus archivos MRI y obtener resultados.

### 2. Ejecutar Scripts individuales (Modo Desarrollo/Prueba)
Si deseas ejecutar etapas específicas del pipeline o realizar un entrenamiento completo, puedes llamar directamente a los scripts. Asegúrate de estar en la raíz del proyecto (MRI_AI_Diagnosis) y con tu entorno virtual activado.

* **Ejemplo de Preprocesamiento DICOM a NIfTI:**
    ```
    python src/data_processing/dicom_to_nifti.py
    ```
    (Asegúrate de ajustar las rutas de entrada/salida dentro del if __name__ == "__main__": del script o pasarlas como argumentos si el script lo soporta).

* **Ejemplo de Registro de máscaras (ANTs):**
    ```
    python src/data_processing/ants_registration.py
    ```
* **Ejemplo de Entrenamiento del UNet:**
    ```
    python src/segmentation/unet_trainer.py
    ```
    (Este script espera que tus datos estén preparados en las rutas definidas en src/config.py y que la carga de datos funcione).

* **Ejemplo de Extracción de características y clasificación (Pipeline Completo):**
    Para ejecutar el pipeline completo de entrenamiento y evaluación tal como se describe en el script original (asumiendo que está actualizado para usar los nuevos módulos), puedes usar el siguiente script `src/main_pipeline.py`.
    ```Python
    # src/main_pipeline.py
    from src.data_processing.data_loader import load_all_data, MultitaskSegmentationDataset, FeatureExtractionDataset, IMAGE_TRANSFORMS
    from src.segmentation.unet_architecture import build_unet_multitask
    from src.segmentation.unet_trainer import train_unet
    from src.segmentation.unet_inference import load_unet_model
    from src.classification.feature_extraction import extract_features
    from src.classification.classifier_model import train_and_evaluate_xgboost, load_xgboost_classifier
    from src.config import DATA_DIR, MASKS_DIR, REGIONS, DEVICE, UNET_MODEL_PATH, XGB_CLASSIFIER_PATH
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report
    
    def run_full_pipeline():
        print("Cargando todos los datos...")
        full_data = load_all_data(DATA_DIR, MASKS_DIR)

    train_data, temp_data = train_test_split(
        full_data, test_size=0.30, stratify=[d['label'] for d in full_data], random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=[d['label'] for d in temp_data], random_state=42
    )

    # 1. Entrenamiento de la U-Net (si no está ya entrenada)
    train_ds = MultitaskSegmentationDataset(train_data, transform=IMAGE_TRANSFORMS)
    val_ds = MultitaskSegmentationDataset(val_data, transform=IMAGE_TRANSFORMS)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    unet_model = build_unet_multitask(len(REGIONS), DEVICE)
    print("🚀 Entrenando U-Net multitarea para segmentación...")
    train_unet(unet_model, train_loader, val_loader)

    # 2. Cargar el UNet (si ya está entrenado o después de entrenar)
    print("Cargando el modelo UNet...")
    trained_unet_model = load_unet_model() # Carga desde UNET_MODEL_PATH

    # 3. Extracción de características
    print("💪 Extrayendo características para clasificación...")
    feature_ds = FeatureExtractionDataset(full_data, transform=IMAGE_TRANSFORMS)
    X_features, y_labels = extract_features(trained_unet_model, feature_ds)

    # 4. Entrenamiento y evaluación de XGBoost
    print("\n🔬 Entrenando y evaluando XGBoostClassifier...")
    best_xgb_model = train_and_evaluate_xgboost(X_features, y_labels) # Guardará el modelo

    # 5. Reporte final
    y_pred_final = best_xgb_model.predict(X_features)
    print("\n--- Reporte de Clasificación Final (XGBoost sobre el conjunto completo de características) ---")
    print(classification_report(y_labels, y_pred_final, digits=4))

    if __name__ == "__main__":
        run_full_pipeline()
    ```
    Puedes guardar este código como `src/main_pipeline.py` y ejecutarlo con:
    
    ```
    python src/main_pipeline.py
    ```

---

## 📂 Datos de ejemplo
Para facilitar el testing y la comprensión del proyecto, se incluyen pequeños conjuntos de datos de ejemplo en la carpeta `data/`:

* `data/raw_dicom_example/`: Una estructura mínima de carpetas que simula un dataset DICOM real.
* `data/registered_nifti_example/`: MRIs de ejemplo que han pasado por la etapa de preprocesamiento y registro.
* `data/segmentation_masks_template/`: Máscaras de segmentación cerebrales de un atlas (ej. MNI) utilizadas como referencia para el registro espacial.

---

## 📝 Licencia
Este proyecto está distribuido bajo la MIT License. Puedes encontrar los detalles completos en el archivo `LICENSE`.
