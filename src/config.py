"""
Configurações globais do projeto Heart Disease Classification.
Centraliza paths, seeds e hiperparâmetros fixos para reprodutibilidade.
"""
from pathlib import Path

# ============================================================
# Paths absolutos baseados na raiz do repositório
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Raw / Processed
RAW_DATA_PATH = DATA_DIR / "raw" / "heart_disease_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "train_test_data.npz"

# Artifacts
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
BEST_MODEL_BASELINE_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_TUNED_PATH = MODELS_DIR / "best_model_tuned.joblib"
TUNING_RESULTS_PATH = MODELS_DIR / "tuning_results.json"

# ============================================================
# Reprodutibilidade e Validação Cruzada
# ============================================================
SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.2

# StratifiedKFold padrão do projeto
CV = {
    "n_splits": N_SPLITS,
    "shuffle": True,
    "random_state": SEED,
}

# ============================================================
# Métricas e Scoring
# ============================================================
# Métrica principal para otimização (CV e tuning)
PRIMARY_SCORING = "roc_auc"

# Métricas para relatório final
REPORT_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

# ============================================================
# Target e Classes
# ============================================================
TARGET_COLUMN = "target"
CLASS_NAMES = ["Saudável (0)", "Doente (1)"]
POSITIVE_CLASS = 1  # Classe de interesse médico (doente)

# ============================================================
# Hiperparâmetros fixos dos modelos baseline (Notebook 04)
# ============================================================
BASELINE_MODELS = {
    "logistic_regression": {
        "class": "LogisticRegression",
        "params": {
            "max_iter": 1000,
            "random_state": SEED,
        },
    },
    "random_forest": {
        "class": "RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "random_state": SEED,
            "n_jobs": -1,
        },
    },
    "decision_tree": {
        "class": "DecisionTreeClassifier",
        "params": {
            "random_state": SEED,
        },
    },
    "knn": {
        "class": "KNeighborsClassifier",
        "params": {
            "n_neighbors": 5,
        },
    },
    "svm": {
        "class": "CalibratedClassifierCV",
        "params": {
            "estimator": {"class": "SVC", "params": {"random_state": SEED}},
        },
    },
}

# ============================================================
# Espaços de busca para Hyperparameter Tuning (Notebook 04.1)
# ============================================================
# Logistic Regression - GridSearchCV (exaustivo)
LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l2", "l1", "elasticnet"],
    "solver": ["lbfgs", "liblinear", "saga"],
    "class_weight": [None, "balanced"],
}

# Random Forest - RandomizedSearchCV (amostragem)
RF_PARAM_DIST = {
    "n_estimators": (100, 500),  # randint
    "max_depth": [None, 5, 10, 15, 20, 25, 30],
    "min_samples_split": (2, 20),  # randint
    "min_samples_leaf": (1, 10),  # randint
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced", "balanced_subsample"],
}

RF_N_ITER = 50

# ============================================================
# Features originais (para referência e inferência)
# ============================================================
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES