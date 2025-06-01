# src/classification/classifier_model.py
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from src.config import XGB_CLASSIFIER_PATH

def train_and_evaluate_xgboost(X: np.ndarray, y: np.ndarray, test_size: float = 0.15,
                                     val_split_ratio: float = 0.1765) -> XGBClassifier:
    """
    Entrena un modelo XGBoostClassifier con ajuste de hiperparámetros
    y lo evalúa en un conjunto de validación.

    Args:
        X (np.ndarray): Conjunto de características.
        y (np.ndarray): Conjunto de etiquetas.
        test_size (float): Proporción del dataset a usar como conjunto de prueba.
        val_split_ratio (float): Proporción del conjunto de entrenamiento a usar
                                 como conjunto de validación.

    Returns:
        XGBClassifier: El modelo XGBoostClassifier entrenado y ajustado.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, stratify=y_resampled, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split_ratio, stratify=y_train, random_state=42
    )

    print("\n🔧 Ajustando hiperparámetros para XGBoost...")
    # Parámetros para GridSearchCV para XGBoost.
    # Ajusta estos según tus necesidades o experimentos previos.
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    # Asegúrate de especificar eval_metric y use_label_encoder para evitar warnings
    xgb = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring='f1_macro', verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_xgb = grid.best_estimator_
    print(f"🏁 Mejor configuración para XGBoost: {grid.best_params_}")

    y_val_pred = best_xgb.predict(X_val)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    acc = accuracy_score(y_val, y_val_pred)
    auc_score = roc_auc_score(y_val, best_xgb.predict_proba(X_val), multi_class='ovr')

    print(f"\n📊 Métricas de validación para XGBoost: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")

    joblib.dump(best_xgb, XGB_CLASSIFIER_PATH)
    print(f"Clasificador XGBoost guardado como '{XGB_CLASSIFIER_PATH}'.")
    return best_xgb
    
    
def load_xgboost_classifier() -> XGBClassifier:
    """Carga el clasificador XGBoost pre-entrenado."""
    try:
        classifier = joblib.load(XGB_CLASSIFIER_PATH)
        print(f"Clasificador XGBoost cargado exitosamente desde {XGB_CLASSIFIER_PATH}.")
    except FileNotFoundError:
        print(f"Error: Clasificador XGBoost no encontrado en {XGB_CLASSIFIER_PATH}. Asegúrate de haberlo entrenado y guardado.")
        raise
    except Exception as e:
        print(f"Error al cargar el Clasificador XGBoost: {e}")
        raise
    return classifier