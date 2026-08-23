"""
Módulo de avaliação e visualização padronizados.
Elimina duplicação de código entre Notebooks 04.1, 05 e 06.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)

from src.config import CLASS_NAMES, POSITIVE_CLASS, MODELS_DIR


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Avalia um classificador no conjunto de teste e retorna métricas padronizadas.

    Args:
        model: Estimator fitted com predict() e predict_proba()
        X_test: Features de teste
        y_test: Labels verdadeiros
        model_name: Nome para identificação no relatório

    Returns:
        Dict com accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_proba),
    }

    return metrics


def evaluate_multiple_models(
    models: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    sort_by: str = "ROC AUC",
) -> pd.DataFrame:
    """
    Avalia múltiplos modelos e retorna DataFrame comparativo ordenado.

    Args:
        models: Dict {nome: estimator_fitted}
        X_test: Features de teste
        y_test: Labels verdadeiros
        sort_by: Coluna para ordenação descendente

    Returns:
        DataFrame com métricas de todos os modelos
    """
    results = []
    for name, model in models.items():
        results.append(evaluate_model(model, X_test, y_test, name))

    df = pd.DataFrame(results)
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return df


def print_classification_report(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Gera e imprime classification_report detalhado.

    Returns:
        String do relatório (também imprime no stdout)
    """
    if target_names is None:
        target_names = CLASS_NAMES

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print(report)
    return report


def plot_confusion_matrix(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Matriz de Confusão",
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
) -> plt.Axes:
    """
    Plota matriz de confusão com seaborn.

    Args:
        model: Estimator fitted
        X_test: Features de teste
        y_test: Labels verdadeiros
        ax: Axes do matplotlib (opcional, cria novo se None)
        title: Título do gráfico
        class_names: Nomes das classes (default: config.CLASS_NAMES)
        normalize: Se True, normaliza por linha (taxa)

    Returns:
        Axes do matplotlib
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        fmt = "d"

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proporção" if normalize else "Contagem"},
    )
    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="both", rotation=0)

    return ax


def plot_roc_curves(
    models: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Curvas ROC",
    reference_line: bool = True,
) -> plt.Axes:
    """
    Plota curvas ROC de múltiplos modelos no mesmo gráfico.

    Args:
        models: Dict {nome: estimator_fitted}
        X_test: Features de teste
        y_test: Labels verdadeiros
        ax: Axes do matplotlib (opcional)
        title: Título do gráfico
        reference_line: Se True, plota diagonal y=x

    Returns:
        Axes do matplotlib
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)

    if reference_line:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Aleatório")

    ax.set_title(title)
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_confusion_matrices_grid(
    models: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_cols: int = 2,
    figsize: Tuple[int, int] = (12, 10),
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
) -> plt.Figure:
    """
    Plota matrizes de confusão de múltiplos modelos em grid.

    Returns:
        Figure do matplotlib
    """
    n_models = len(models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, model) in enumerate(models.items()):
        plot_confusion_matrix(
            model, X_test, y_test, ax=axes[idx],
            title=f"Matriz de Confusão — {name}",
            class_names=class_names,
            normalize=normalize,
        )

    # Esconde axes extras
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def get_feature_importance_or_coefs(
    model: BaseEstimator,
    feature_names: np.ndarray,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Extrai feature_importances_ (tree-based) ou coef_ (linear models).

    Para Logistic Regression, retorna Odds Ratio = exp(coef).

    Args:
        model: Estimator fitted
        feature_names: Nomes das features (preprocessor.get_feature_names_out())
        top_k: Número de top features para retornar

    Returns:
        DataFrame com colunas ['feature', 'importance'] ordenado desc
    """
    if hasattr(model, "feature_importances_"):
        # Tree-based (Random Forest, Decision Tree, etc.)
        importances = model.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": importances})

    elif hasattr(model, "coef_"):
        # Linear models (Logistic Regression, Linear SVM, etc.)
        # coef_ shape: (n_classes, n_features) para multiclass, (n_features,) para binary
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]  # binary classification -> pega classe positiva
        # Odds Ratio para interpretabilidade médica
        odds_ratio = np.exp(coef)
        df = pd.DataFrame({"feature": feature_names, "importance": odds_ratio})
        df["coef_raw"] = coef  # mantém coeficiente original para referência

    else:
        raise AttributeError(
            f"Modelo {type(model).__name__} não possui feature_importances_ nem coef_"
        )

    df = df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
    return df


def plot_feature_importance(
    model: BaseEstimator,
    feature_names: np.ndarray,
    top_k: int = 15,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    palette: str = "viridis",
) -> plt.Axes:
    """
    Plota barplot das top-k features mais importantes.

    Para LR, plota Odds Ratio (exp(coef)). Para RF, feature_importances_.
    """
    df = get_feature_importance_or_coefs(model, feature_names, top_k=top_k)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))

    is_lr = hasattr(model, "coef_")
    xlabel = "Odds Ratio (exp(coef))" if is_lr else "Importância (Gini)"

    sns.barplot(data=df, x="importance", y="feature", palette=palette, ax=ax, orient="h")
    ax.set_title(title or f"Top {top_k} Features — {type(model).__name__}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.invert_yaxis()  # maior importância no topo

    # Adiciona valores nas barras
    for i, (_, row) in enumerate(df.iterrows()):
        val = row["importance"]
        ax.text(val + 0.01 * df["importance"].max(), i, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return ax


def save_evaluation_report(
    metrics_df: pd.DataFrame,
    best_model_name: str,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Salva relatório de avaliação em JSON para auditoria.

    Args:
        metrics_df: DataFrame retornado por evaluate_multiple_models
        best_model_name: Nome do modelo selecionado
        output_path: Path customizado (default: models/evaluation_report.json)

    Returns:
        Path do arquivo salvo
    """
    if output_path is None:
        output_path = MODELS_DIR / "evaluation_report.json"

    report = {
        "best_model": best_model_name,
        "test_metrics": metrics_df.to_dict(orient="records"),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return output_path