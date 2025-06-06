# src/gui/app.py
import streamlit as st
import nibabel as nib
import numpy as np
from pathlib import Path
import os
import pandas as pd
import tempfile # Importar tempfile para manejar archivos temporales
import sys

# --- Configuración del entorno y rutas ---
# Obtiene la ruta del directorio del script actual (src/gui)
current_dir = Path(__file__).resolve().parent
# Obtiene la ruta de la raíz del proyecto (la carpeta padre de 'src')
# Si tu proyecto es ruta/a/MRI_AI_Diagnosis/src/gui/app.py
# current_dir será ruta/a/MRI_AI_Diagnosis/src/gui
# current_dir.parent será ruta/a/MRI_AI_Diagnosis/src
# current_dir.parent.parent será ruta/a/MRI_AI_Diagnosis (la raíz del proyecto)
project_root = current_dir.parent.parent 

# Añade la raíz del proyecto al sys.path
# Esto permite que Python encuentre el paquete 'src'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DEVICE, REGIONS, REGION_FULL_NAMES, IMAGE_SIZE, CLASS_LABELS
from src.segmentation.unet_inference import load_unet_model, preprocess_mri, perform_segmentation
from src.classification.classifier_model import load_xgboost_classifier
from src.classification.feature_extraction import extract_region_volumes, extract_shape_features, extract_texture_features, get_feature_names

# --- Configuración Global ---

# --- Funciones de Carga de Modelos ---
@st.cache_resource # Cacha el modelo para que no se recargue en cada interacción de Streamlit
def get_unet_model():
    """Carga y cachea el modelo UNet."""
    return load_unet_model()

@st.cache_resource # Cacha el clasificador para que no se recargue en cada interacción de Streamlit
def get_xgboost_classifier():
    """Carga y cachea el clasificador XGBoost."""
    return load_xgboost_classifier()

# --- Datos de referencia hipotéticos para la explicación clínica ---
# Estos datos deberían ser obtenidos de un conjunto de datos de entrenamiento representativo
# y validados clínicamente para un uso real. Aquí son solo para demostración.
def generate_mock_reference_ranges(feature_names, regions):
    """
    Genera rangos de referencia hipotéticos para varias características
    para las clases 'Alzheimer', 'Parkinson' y 'Control'.
    Estos datos son solo para fines de demostración y no son médicamente precisos.

    Args:
        feature_names (list[str]): Nombres de las características.
        regions (list[str]): Nombres de las regiones de interés.

    Returns:
        dict: Un diccionario anidado con los rangos de referencia (media, desviación estándar)
              para cada característica y clase.
    """
    mock_ranges = {
        "Alzheimer": {},
        "Parkinson": {},
        "Control": {}
    }

    # Valores base para Control (saludable)
    base_control_volumes = {
        "left_hippocampus": 1400, "right_hippocampus": 1450,
        "left_putamen": 1800, "right_putamen": 1850
    }
    base_control_area_2d = 200
    base_control_eccentricity = 0.7
    base_control_solidity = 0.95
    base_control_contrast = 0.08
    base_control_homogeneity = 0.98
    base_control_energy = 0.85
    base_control_correlation = 0.95

    for feature_name in feature_names:
        # Determinar tipo de característica y región
        parts = feature_name.split(' - ')
        region_full_name = parts[0]
        feature_type = parts[1]
        
        region_key = next(k for k, v in REGION_FULL_NAMES.items() if v == region_full_name)

        # Control
        if "Volumen" in feature_type:
            mean_c = base_control_volumes[region_key]
            std_c = mean_c * 0.07 # 7% std dev
            mock_ranges["Control"][feature_name] = (mean_c, std_c)
            # Alzheimer: volúmenes más pequeños (ej., 25% de reducción para hipocampo, 10% para putamen)
            mean_a = mean_c * 0.75 if "hippocampus" in region_key else mean_c * 0.9
            std_a = std_c * 1.2
            mock_ranges["Alzheimer"][feature_name] = (mean_a, std_a)
            # Parkinson: el putamen podría estar más afectado, el hipocampo menos que en Alzheimer
            mean_p = mean_c * 0.95 if "hippocampus" in region_key else mean_c * 0.85
            std_p = std_c * 1.1
            mock_ranges["Parkinson"][feature_name] = (mean_p, std_p)
        elif "Área 2D" in feature_type:
            mean_c = base_control_area_2d
            std_c = mean_c * 0.1
            mock_ranges["Control"][feature_name] = (mean_c, std_c)
            mean_a = mean_c * 0.75
            std_a = std_c * 1.2
            mock_ranges["Alzheimer"][feature_name] = (mean_a, std_a)
            mean_p = mean_c * 0.9
            std_p = std_c * 1.1
            mock_ranges["Parkinson"][feature_name] = (mean_p, std_p)
        elif "Excentricidad" in feature_type: # Mayor para formas más alargadas
            mean_c = base_control_eccentricity
            std_c = 0.03
            mock_ranges["Control"][feature_name] = (mean_c, std_c)
            mean_a = mean_c * 1.2 # Más excéntrico para Alzheimer (más alargado)
            std_a = std_c * 1.5
            mock_ranges["Alzheimer"][feature_name] = (mean_a, std_a)
            mean_p = mean_c * 1.05 # Ligeramente más excéntrico para Parkinson
            std_p = std_c * 1.2
            mock_ranges["Parkinson"][feature_name] = (mean_p, std_p)
        elif "Solidez" in feature_type: # Menor para formas más irregulares
            mean_c = base_control_solidity
            std_c = 0.01
            mock_ranges["Control"][feature_name] = (mean_c, std_c)
            mean_a = mean_c * 0.9 # Menos sólido para Alzheimer (más irregular)
            std_a = std_c * 2
            mock_ranges["Alzheimer"][feature_name] = (mean_a, std_a)
            mean_p = mean_c * 0.95 # Ligeramente menos sólido para Parkinson
            std_p = std_c * 1.5
            mock_ranges["Parkinson"][feature_name] = (mean_p, std_p)
        elif "Contraste" in feature_type: # Mayor para tejido más heterogéneo
            mean_c = base_control_contrast
            std_c = 0.01
            mock_ranges["Control"][feature_name] = (mean_c, std_c)
            mean_a = mean_c * 1.5 # Mayor contraste para Alzheimer (más heterogeneidad)
            std_a = std_c * 2
            mock_ranges["Alzheimer"][feature_name] = (mean_a, std_a)
            mean_p = mean_c * 1.2
            std_p = std_c * 1.5
            mock_ranges["Parkinson"][feature_name] = (mean_p, std_p)
        elif "Homogeneidad" in feature_type: # Menor para tejido menos uniforme
            mean_c = base_control_homogeneity
            std_c = 0.005
            mock_ranges["Control"][feature_name] = (mean_c, std_c)
            mean_a = mean_c * 0.8 # Menor homogeneidad para Alzheimer
            std_a = std_c * 2
            mock_ranges["Alzheimer"][feature_name] = (mean_a, std_a)
            mean_p = mean_c * 0.9
            std_p = std_c * 1.5
            mock_ranges["Parkinson"][feature_name] = (mean_p, std_p)
        elif "Energía" in feature_type: # Menor para tejido menos uniforme
            mean_c = base_control_energy
            std_c = 0.03
            mock_ranges["Control"][feature_name] = (mean_c, std_c)
            mean_a = mean_c * 0.7 # Menor energía para Alzheimer
            std_a = std_c * 2
            mock_ranges["Alzheimer"][feature_name] = (mean_a, std_a)
            mean_p = mean_c * 0.8
            std_p = std_c * 1.5
            mock_ranges["Parkinson"][feature_name] = (mean_p, std_p)
        elif "Correlación" in feature_type: # Menor para patrones menos estructurados
            mean_c = base_control_correlation
            std_c = 0.02
            mock_ranges["Control"][feature_name] = (mean_c, std_c)
            mean_a = mean_c * 0.8 # Menor correlación para Alzheimer
            std_a = std_c * 1.5
            mock_ranges["Alzheimer"][feature_name] = (mean_a, std_a)
            mean_p = mean_c * 0.9
            std_p = std_c * 1.2
            mock_ranges["Parkinson"][feature_name] = (mean_p, std_p)
    return mock_ranges

HYPOTHETICAL_REFERENCE_RANGES = generate_mock_reference_ranges(get_feature_names(), REGIONS)


# --- GUI - Diseño y Lógica con Streamlit ---

st.set_page_config(
    page_title="MRI AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Herramienta de soporte al diagnóstico de RM Cerebral")
st.markdown("---")

# Cargar modelos una sola vez al inicio de la aplicación
unet_model = get_unet_model()
xgb_classifier = get_xgboost_classifier()

FEATURE_NAMES = get_feature_names()

# Sección para subir archivo en la barra lateral
with st.sidebar:
    st.header("Cargar MRI del Paciente")
    uploaded_file = st.file_uploader(
        "Sube un archivo NIfTI (.nii o .nii.gz) de MRI:", type=["nii", "nii.gz"]
    )

# Sección principal para visualización y resultados
col1, col2 = st.columns([0.5, 0.5])

with col1:
    st.header("Visualización de la MRI")
    if uploaded_file is not None:
        temp_file_path = None # Inicializar a None

        try:
            # Leer el archivo NIfTI
            bytes_data = uploaded_file.getvalue()
            # Crear un archivo temporal para nibabel
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
                tmp_file.write(bytes_data)
                temp_file_path = Path(tmp_file.name) # Guardar la ruta del archivo temporal

            mri_img = nib.load(temp_file_path)
            mri_data_original = mri_img.get_fdata()

            # Almacenar los datos originales en session_state para persistencia y navegación
            st.session_state['mri_data_original'] = mri_data_original

            # Generar y mostrar ID del paciente desde el nombre del archivo
            # Solo generar nuevo ID si el archivo subido es diferente al anterior
            if 'patient_id' not in st.session_state or st.session_state.get('last_uploaded_file_name') != uploaded_file.name:
                st.session_state['patient_id'] = Path(uploaded_file.name).stem # Usar el nombre del archivo como ID
                st.session_state['last_uploaded_file_name'] = uploaded_file.name # Rastrear el último archivo subido
                # Resetear la clasificación si se sube un nuevo archivo
                st.session_state.pop('predicted_label', None)
                st.session_state.pop('extracted_features', None)
                st.session_state.pop('prediction_probs', None)
                st.session_state.pop('segmented_masks_tensor', None) # Limpiar máscaras anteriores
                # Resetear el índice de corte a la mitad si es un nuevo archivo
                st.session_state['current_slice_idx'] = mri_data_original.shape[2] // 2


            st.subheader(f"MRI cargada para ID: **{st.session_state['patient_id']}**")

            # Inicializar el índice de corte si no está ya en session_state
            if 'current_slice_idx' not in st.session_state:
                st.session_state['current_slice_idx'] = mri_data_original.shape[2] // 2

            # Índice máximo del corte
            max_slice_idx = mri_data_original.shape[2] - 1

            # Funciones para sincronizar el slider y el number_input
            def update_slice_from_slider():
                st.session_state['current_slice_idx'] = st.session_state['slice_slider_key']

            def update_slice_from_number_input():
                st.session_state['current_slice_idx'] = st.session_state['slice_number_input_key']
            
            # Crear columnas para el textbox y el slider
            slice_col1, slice_col2 = st.columns(2)

            with slice_col1:
                # Entrada numérica para el corte
                st.number_input(
                    "Número de Corte Axial",
                    min_value=0,
                    max_value=max_slice_idx,
                    value=st.session_state['current_slice_idx'],
                    key="slice_number_input_key",
                    on_change=update_slice_from_number_input,
                    help="Introduce el número de corte axial para visualizar."
                )
            with slice_col2:
                # Slider para navegar por los cortes
                st.slider(
                    "Seleccionar Corte Axial",
                    0,
                    max_slice_idx,
                    value=st.session_state['current_slice_idx'],
                    key="slice_slider_key",
                    on_change=update_slice_from_slider,
                    help="Usa el slider para navegar a través de los cortes axiales."
                )
            
            # El índice de corte a mostrar es el que está en session_state
            slice_idx_to_display = st.session_state['current_slice_idx']

            # Preparar corte para visualización basada en el slider/number_input
            mri_slice_display = mri_data_original[:, :, slice_idx_to_display]

            # Normalizar para visualización
            mri_slice_display = (mri_slice_display - mri_slice_display.min()) / (mri_slice_display.max() - mri_slice_display.min() + 1e-8)
            
            # Usar use_container_width en lugar de use_column_width
            st.image(np.rot90(mri_slice_display), caption=f"MRI Original (Corte Axial {slice_idx_to_display + 1}/{mri_data_original.shape[2]})", use_container_width=True)

            st.header("Resultados de la clasificación")
            
            # --- Clasificación Automática ---
            # Solo clasificar si no se ha clasificado ya para este archivo
            if 'predicted_label' not in st.session_state:
                with st.spinner("Procesando y clasificando automáticamente... Esto puede tomar un momento."):
                    # Preprocesar la MRI para el UNet
                    preprocessed_mri_tensor = preprocess_mri(mri_data_original)

                    # Realizar segmentación
                    segmented_masks_tensor = perform_segmentation(unet_model, preprocessed_mri_tensor)
                    st.session_state['segmented_masks_tensor'] = segmented_masks_tensor # Almacenar máscaras segmentadas

                    # Extraer características
                    volumes = extract_region_volumes(segmented_masks_tensor)
                    shape_features = extract_shape_features(segmented_masks_tensor)
                    texture_features = extract_texture_features(segmented_masks_tensor)
                    
                    extracted_features = np.array(volumes + shape_features + texture_features).reshape(1, -1)

                    # Clasificar con XGBoost
                    prediction_probs = xgb_classifier.predict_proba(extracted_features)[0]
                    predicted_class_idx = np.argmax(prediction_probs)
                    predicted_label = CLASS_LABELS[predicted_class_idx]

                    # Guardar resultados en st.session_state para la explicación
                    st.session_state['predicted_label'] = predicted_label
                    st.session_state['prediction_probs'] = prediction_probs
                    st.session_state['extracted_features'] = extracted_features[0] # Almacenar como 1D array
                    # El ID del paciente ya está en session_state
            
            # Mostrar resultados de clasificación (ya sea automática o de una sesión anterior)
            if 'predicted_label' in st.session_state:
                current_patient_id = st.session_state.get('patient_id', 'N/A')
                predicted_label = st.session_state['predicted_label']
                prediction_probs = st.session_state['prediction_probs']

                st.markdown(f"## **Diagnóstico Predicho para ID {current_patient_id}: <span style='color:green;'>{predicted_label}</span>**", unsafe_allow_html=True)
                st.write("Probabilidades de clase:")
                for i, prob in enumerate(prediction_probs):
                    st.write(f"- {CLASS_LABELS[i]}: {prob:.2%}")

        except Exception as e:
            st.error(f"Se produjo un error al procesar el archivo: {e}")
            st.warning("Asegúrate de que el archivo es un NIfTI 3D válido con las dimensiones esperadas por el modelo.")
            # Limpiar estado si hay un error al procesar el archivo
            st.session_state.pop('mri_data_original', None)
            st.session_state.pop('patient_id', None)
            st.session_state.pop('last_uploaded_file_name', None)
            st.session_state.pop('current_slice_idx', None)
            st.session_state.pop('predicted_label', None)
            st.session_state.pop('extracted_features', None)
            st.session_state.pop('prediction_probs', None)
            st.session_state.pop('segmented_masks_tensor', None)
        finally:
            # Asegurarse de eliminar el archivo temporal incluso si hay un error
            if temp_file_path and temp_file_path.exists():
                os.unlink(temp_file_path)

    else:
        st.info("Por favor, sube un archivo MRI para visualizarlo y clasificarlo.")

with col2:
    st.header("Explicación del modelo")
    # Usar st.session_state para mantener el estado de la predicción y las características
    if 'predicted_label' in st.session_state and 'extracted_features' in st.session_state:
        predicted_label = st.session_state['predicted_label']
        extracted_features_patient = st.session_state['extracted_features'] # Características del paciente
        patient_id_display = st.session_state.get('patient_id', 'N/A') # Recuperar ID del paciente
        prediction_probs = st.session_state['prediction_probs'] # Recuperar probabilidades

        st.subheader(f"Explicación para el diagnóstico de **{predicted_label}** (ID: {patient_id_display})")

        # Obtener importancias de las características del clasificador XGBoost
        feature_importances = xgb_classifier.feature_importances_

        # Crear un DataFrame para visualizar mejor las importancias y los valores del paciente
        explanation_df = pd.DataFrame({
            'Característica': FEATURE_NAMES,
            'Importancia': feature_importances,
            'Valor del Paciente': extracted_features_patient
        })
        
        # Añadir columnas para los rangos de referencia
        explanation_df['Rango Típico (Control)'] = explanation_df['Característica'].apply(
            lambda x: f"{HYPOTHETICAL_REFERENCE_RANGES['Control'][x][0]:.2f} ± {HYPOTHETICAL_REFERENCE_RANGES['Control'][x][1]:.2f}"
            if x in HYPOTHETICAL_REFERENCE_RANGES['Control'] else 'N/A'
        )
        explanation_df['Rango Típico (Alzheimer)'] = explanation_df['Característica'].apply(
            lambda x: f"{HYPOTHETICAL_REFERENCE_RANGES['Alzheimer'][x][0]:.2f} ± {HYPOTHETICAL_REFERENCE_RANGES['Alzheimer'][x][1]:.2f}"
            if x in HYPOTHETICAL_REFERENCE_RANGES['Alzheimer'] else 'N/A'
        )
        explanation_df['Rango Típico (Parkinson)'] = explanation_df['Característica'].apply(
            lambda x: f"{HYPOTHETICAL_REFERENCE_RANGES['Parkinson'][x][0]:.2f} ± {HYPOTHETICAL_REFERENCE_RANGES['Parkinson'][x][1]:.2f}"
            if x in HYPOTHETICAL_REFERENCE_RANGES['Parkinson'] else 'N/A'
        )

        # Ordenar por importancia y mostrar las top N
        top_n_features = 10 
        explanation_df = explanation_df.sort_values(by='Importancia', ascending=False).head(top_n_features)

        st.markdown(f"""
        El modelo ha clasificado esta resonancia magnética como **{predicted_label}** con las siguientes probabilidades:
        """)
        for i, prob in enumerate(prediction_probs):
            st.write(f"- {CLASS_LABELS[i]}: {prob:.2%}")

        st.markdown(f"""
        Esta clasificación se basa en el análisis de diversas características (volumen, forma y textura) extraídas de regiones cerebrales específicas.
        A continuación, se muestran las **{top_n_features} características que el modelo consideró más influyentes** para llegar a este diagnóstico, junto con los valores medidos en la MRI del paciente y rangos de referencia hipotéticos para cada clase:
        """)
        st.table(explanation_df)

        st.markdown("""
        **Interpretación de las características clave:**
        Los valores de 'Importancia' indican cuánto contribuyó cada característica a la decisión final del modelo. Un valor más alto significa una mayor influencia. El modelo ha aprendido patrones específicos en estas características que se asocian con cada una de las clases (Alzheimer, Parkinson, Control).

        Para cada característica listada, puede comparar el **'Valor del Paciente'** con los **'Rangos Típicos'** de las clases de referencia. Por ejemplo:
        * Si el diagnóstico predicho es **Alzheimer**, observe si los valores del paciente para características como el **volumen del hipocampo** son consistentemente **inferiores** a los rangos de control y más cercanos a los rangos típicos de Alzheimer.
        * Si el diagnóstico predicho es **Parkinson**, preste atención a las características del **putamen**, como su **volumen** o **propiedades de textura**, y cómo se comparan con los rangos típicos de Parkinson.
        * Para un diagnóstico de **Control**, los valores del paciente deberían estar generalmente dentro de los rangos típicos de la población de control.

        Las desviaciones significativas del 'Rango Típico (Control)' hacia los rangos de las clases de enfermedad, especialmente para características con alta 'Importancia', son las que el modelo utiliza como base para su predicción.
        """)

        st.markdown("""
        ### Consideraciones clínicas y validez del modelo:
        Esta herramienta es un **soporte al diagnóstico** y no debe interpretarse como un diagnóstico médico definitivo. La validez clínica de este modelo se basa en la calidad y representatividad del conjunto de datos con el que fue entrenado y validado.

        **Para evaluar la validez clínica en este caso específico, el médico debe considerar:**
        * **Probabilidad de predicción:** Una probabilidad alta para la clase predicha (ej., 90% para Alzheimer) indica una alta confianza del modelo en su clasificación para este paciente. Probabilidades más bajas (ej., 55%) sugieren mayor incertidumbre y requieren una revisión más crítica.
        * **Coherencia con la clínica:** Los hallazgos del modelo (características influyentes y sus valores) deben ser coherentes con la presentación clínica del paciente, su historial médico y otros resultados de pruebas.
        * **Análisis de rangos de referencia:** La comparación de los valores del paciente con los rangos de referencia proporcionados (que idealmente provendrían de un estudio poblacional robusto) ayuda a contextualizar las mediciones del paciente. Si los valores del paciente se alinean bien con los patrones conocidos de la enfermedad predicha, esto refuerza la confianza.
        * **Limitaciones del modelo:** Reconozca que los modelos de IA tienen limitaciones y pueden no capturar todas las complejidades biológicas.
        
        La decisión final siempre debe ser tomada por un profesional de la salud cualificado, integrando esta información con su juicio clínico experto.
        """)
    else:
        st.info("Sube una MRI para visualizarla y ver la clasificación y explicación.")

st.markdown("---")
st.sidebar.markdown("""
    ### Notas:
    * Este prototipo utiliza un modelo de aprendizaje automático para el soporte al diagnóstico. No debe reemplazar el juicio clínico profesional.
    * Asegúrate de que los archivos `unet_multitask.pth` y `xgboost_classifier.joblib` están en el mismo directorio que este script.
    * El modelo UNet espera imágenes con un tamaño de (96, 96, 96) después del preprocesamiento.
""")