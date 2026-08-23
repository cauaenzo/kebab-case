"""
Módulo de carregamento de dados e artefatos do projeto.
Fornece funções únicas para evitar duplicação entre notebooks.
"""
import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_PATH,
    PREPROCESSOR_PATH,
    BEST_MODEL_BASELINE_PATH,
    BEST_MODEL_TUNED_PATH,
    TUNING_RESULTS_PATH,
    CLASS_NAMES,
    POSITIVE_CLASS,
)


def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega os arrays de treino e teste pré-processados (Notebook 03).

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo processado não encontrado: {PROCESSED_DATA_PATH}. "
            "Execute o Notebook 03 primeiro."
        )

    data = np.load(PROCESSED_DATA_PATH)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    return X_train, X_test, y_train, y_test


def load_preprocessor():
    """
    Carrega o ColumnTransformer fitted (Notebook 03).

    Returns:
        ColumnTransformer: Preprocessor pronto para .transform()
    """
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor não encontrado: {PREPROCESSOR_PATH}. "
            "Execute o Notebook 03 primeiro."
        )
    return joblib.load(PREPROCESSOR_PATH)


def load_model_artifacts(prefer_tuned: bool = True):
    """
    Carrega modelo e preprocessor com fallback automático.

    Args:
        prefer_tuned: Se True, tenta carregar best_model_tuned.joblib primeiro.
                      Se não existir, cai para best_model.joblib (baseline).

    Returns:
        Tuple: (model, preprocessor, model_source)
               model_source = "tuned" | "baseline"
    """
    preprocessor = load_preprocessor()

    if prefer_tuned and BEST_MODEL_TUNED_PATH.exists():
        model = joblib.load(BEST_MODEL_TUNED_PATH)
        source = "tuned"
    elif BEST_MODEL_BASELINE_PATH.exists():
        model = joblib.load(BEST_MODEL_BASELINE_PATH)
        source = "baseline"
    else:
        raise FileNotFoundError(
            f"Nenhum modelo encontrado. Verifique {BEST_MODEL_BASELINE_PATH} "
            f"ou {BEST_MODEL_TUNED_PATH}. Execute Notebook 04 ou 04.1."
        )

    return model, preprocessor, source


def load_tuning_results() -> dict:
    """
    Carrega metadados da otimização de hiperparâmetros (Notebook 04.1).

    Returns:
        Dict com best_params, best_cv_score, cv_results, selected_model
    """
    if not TUNING_RESULTS_PATH.exists():
        return {}

    with open(TUNING_RESULTS_PATH, "r") as f:
        return json.load(f)


def get_feature_names(preprocessor) -> np.ndarray:
    """
    Extrai nomes das features após transformação (OneHotEncoder expande categóricas).

    Args:
        preprocessor: ColumnTransformer fitted

    Returns:
        Array com nomes de todas as features transformadas
    """
    return preprocessor.get_feature_names_out()


def prepare_inference_dataframe(raw_dict: dict) -> pd.DataFrame:
    """
    Converte dicionário bruto de paciente para DataFrame com colunas na ordem correta.

    Args:
        raw_dict: Dict com keys = features originais (13 features)

    Returns:
        DataFrame com uma linha, colunas ordenadas como no treino
    """
    from src.config import ALL_FEATURES

    df = pd.DataFrame([raw_dict])
    # Garante ordem e presença de todas as colunas
    df = df.reindex(columns=ALL_FEATURES)
    return df