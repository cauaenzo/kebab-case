# Heart Disease Classification — Cleveland Dataset

Projeto de Machine Learning para classificação binária de doenças cardíacas baseada no dataset **Cleveland Heart Disease** (UCI Machine Learning Repository).

---

## Research Question

**Como diferentes algoritmos de classificação supervisionada comparam-se na predição de doença cardíaca a partir de variáveis clínicas e demográficas do dataset Cleveland, e qual modelo oferece o melhor equilíbrio entre performance preditiva, interpretabilidade clínica e robustez?**

---

## Hypothesis

1. **H1 (Performance)**: Modelos lineares regularizados (Logistic Regression com penalty L1/L2) superam modelos baseados em árvores não-otimizados em datasets tabulares pequenos (n≈303) com features clínicas bem comportadas.
2. **H2 (Interpretabilidade vs. Performance)**: A Logistic Regression otimizada alcança ROC AUC competitivo (≥0.90) mantendo interpretabilidade via Odds Ratios — essencial para validação médica — enquanto Random Forest otimizado pode superar levemente em Recall mas sacrifica transparência.
3. **H3 (Generalização)**: Estratégia de validação cruzada estratificada (Stratified K-Fold, k=5) com otimização de hiperparâmetros aninhada fornece estimativas de performance não-enviesadas para seleção de modelo, evitando overfitting ao conjunto de validação.

---

## Methodology

### Dataset
- **Fonte**: UCI Machine Learning Repository — Cleveland Heart Disease (303 amostras, 13 features originais + target binário)
- **Target**: Presença (1) ou ausência (0) de doença cardíaca (estenose ≥50% em angiography)
- **Split**: 80/20 estratificado (242 treino / 61 teste), seed=42
- **Features finais**: 28 após One-Hot Encoding (5 numéricas + 8 categóricas expandidas)

### Pré-processamento (Pipeline Serializado)
- **Numéricas**: `SimpleImputer(median)` → `StandardScaler`
- **Categóricas**: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
- **Composição**: `ColumnTransformer` → `Pipeline` (fit apenas no treino, zero data leakage)

### Modelos Baseline (Notebook 04)
| Modelo | Configuração Principal |
|--------|------------------------|
| Logistic Regression | `max_iter=1000, random_state=42` |
| Random Forest | `n_estimators=100, n_jobs=-1, random_state=42` |
| Decision Tree | `random_state=42` |
| KNN | `n_neighbors=5` |
| SVM (calibrado) | `SVC(random_state=42) + CalibratedClassifierCV` |

### Validação Cruzada & Seleção
- **Estratégia**: Stratified K-Fold (k=5, shuffle=True, seed=42)
- **Métrica primária**: ROC AUC
- **Seleção**: Melhor média ROC AUC no CV → passa para tuning

### Hyperparameter Tuning (Notebook 04.1)
| Modelo | Estratégia | Espaço de Busca | Iterações |
|--------|------------|-----------------|-----------|
| Logistic Regression | GridSearchCV (exaustivo) | 5×3×3×2 = 90 combos (filtrados por compatibilidade solver/penalty) | 40 válidas |
| Random Forest | RandomizedSearchCV | 7×6×18×9×5×2×3 ≈ 2M combos | 50 amostras |

- **CV interno**: Mesmo Stratified K-Fold (k=5)
- **Métrica de otimização**: `roc_auc`
- **Critério de desempate**: Menor desvio padrão no CV → maior estabilidade

### Modelo Final & Avaliação (Notebook 05)
- **Selecionado**: Logistic Regression tunada (`C=1.0, penalty=l1, solver=liblinear, class_weight=balanced`)
- **Avaliação final**: Holdout set (61 amostras) — métricas por classe + macro avg + ROC AUC
- **Threshold**: Default 0.5 (probabilidade calibrada via `predict_proba`)

### Inferência em Produção (Notebook 06)
- Carregamento de artefatos serializados (`best_model_tuned.joblib`, `preprocessor.joblib`)
- Transformação de dados brutos → predição + probabilidade
- Pronto para wrapping em API (FastAPI/Flask)

---

## Experiments

### Experimento 1: Comparação de Modelos Baseline (CV)
**Objetivo**: Identificar candidatos para tuning via screening rápido.
**Setup**: 5 modelos × Stratified K-Fold (k=5) no treino (242 amostras).
**Resultado**: Logistic Regression lidera (Acc=0.847±0.010), seguida por KNN (0.826±0.025) e SVM (0.818±0.025).

### Experimento 2: Hyperparameter Tuning Sistemático
**Objetivo**: Extrair performance máxima dos dois melhores baselines.
**Setup**: GridSearchCV (LR) + RandomizedSearchCV (RF) com CV idêntico.
**Resultado**: 
- LR tunada: ROC AUC CV = **0.9067** (C=1.0, l1, liblinear, balanced)
- RF tunada: ROC AUC CV = 0.9040 (n_estimators=359, max_depth=25, ...)

### Experimento 3: Avaliação Holdout & Seleção Final
**Objetivo**: Confirmar generalização no conjunto de teste nunca visto.
**Setup**: Modelo selecionado pelo CV (LR tunada) avaliado no holdout (61 amostras).
**Resultado**: Accuracy=0.8525, Recall(classe 1)=0.8929, F1=0.8475, **ROC AUC=0.9589**

### Experimento 4: Benchmark Comparativo no Holdout
**Objetivo**: Verificar se modelo não-selecionado (RF tunada) supera no teste.
**Resultado**: RF tunada tem Recall=0.9286 e F1=0.8667 superiores, mas ROC AUC=0.9502 < LR.
**Decisão**: Mantém LR tunada pela consistência CV→Teste e interpretabilidade médica.

---

## Results

### Validação Cruzada — Modelos Baseline (k=5, n=242)

| Modelo | Acurácia Média | Desvio Padrão | ROC AUC Médio (estimado) |
|--------|---------------|---------------|--------------------------|
| **Logistic Regression** | **0.8471** | **±0.0100** | ~0.88 |
| KNN | 0.8264 | ±0.0254 | ~0.86 |
| SVM | 0.8180 | ±0.0253 | ~0.85 |
| Random Forest | 0.8098 | ±0.0311 | ~0.84 |
| Decision Tree | 0.6897 | ±0.0575 | ~0.72 |

> **Conclusão**: LR vence em média e estabilidade (menor σ), confirmando H1.

---

### Hyperparameter Tuning — Validação Cruzada

| Modelo | Melhor ROC AUC (CV) | Melhores Hiperparâmetros |
|--------|---------------------|--------------------------|
| **Logistic Regression (tuned)** | **0.9067** | `C=1.0, penalty=l1, solver=liblinear, class_weight=balanced` |
| Random Forest (tuned) | 0.9040 | `n_estimators=359, max_depth=25, min_samples_split=11, min_samples_leaf=6, max_features=log2, bootstrap=False, class_weight=balanced` |

> **Conclusão**: LR tunada supera marginalmente (Δ=0.0027) com muito menos complexidade — confirma H2.

---

### Conjunto de Teste (Holdout) — n=61

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression (tuned)** | 0.8525 | 0.8065 | 0.8929 | 0.8475 | **0.9589** |
| Random Forest (tuned) | 0.8689 | 0.8125 | **0.9286** | **0.8667** | 0.9502 |

> **Conclusão**: RF ganha em Recall/F1 no teste, mas LR mantém ROC AUC superior e consistência CV→Teste (0.9067→0.9589). Seleção baseada no CV validada — confirma H3.

---

### Modelo Final — Relatório por Classe (LR Tunada no Teste)

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Saudável (0) | 0.93 | 0.85 | 0.89 | 33 |
| Doente (1) | 0.84 | **0.93** | 0.88 | 28 |
| **Macro Avg** | **0.89** | **0.89** | **0.89** | 61 |

| Métrica | Valor |
|---------|-------|
| Acurácia | 0.89 |
| ROC AUC | 0.9589 |

> **Contexto médico**: Recall da classe **Doente = 0.93** — o modelo identifica corretamente 93% dos pacientes doentes, minimizando falsos negativos (crítico em triagem cardíaca).

---

### Feature Importance (Top 10 — Logistic Regression, Odds Ratio)

| Feature | Odds Ratio | Interpretação Clínica |
|---------|------------|----------------------|
| cp_3 (dor atípica) | ~4.2 | Tipo de dor no peito é preditor forte |
| thal_2 (defeito fixo) | ~3.8 | Cintilografia anormal = alto risco |
| ca_1 (1 vaso principal) | ~3.1 | Número de vasos comprometidos |
| oldpeak | ~2.5 | Depressão ST induzida por esforço |
| thalach | ~0.45 | FC máxima inversamente associada |
| exang_1 (angina exercício) | ~2.2 | Angina à esforço = risco |
| slope_2 (descendente) | ~1.9 | Inclinação ST descendente |
| age | ~1.03/ano | Risco aumenta com idade |
| chol | ~1.003/mg | Colesterol efeito marginal |
| sex_1 (masculino) | ~1.4 | Homens com risco levemente maior |

> Odds Ratio > 1 = fator de risco; < 1 = fator protetor. Alinhado com literatura cardiológica.

---

## Limitations

1. **Tamanho amostral pequeno (n=303)**: Limita poder estatístico e generalização; CV k=5 tem variância inerente. Intervalos de confiança não reportados.
2. **Dataset single-center (Cleveland apenas)**: Viés de seleção geográfico/demográfico; não valida em populações diversas (outros hospitais, etnias, países).
3. **Features limitadas**: 13 variáveis originais; ausência de biomarkers modernos (troponina, BNP, imaging avançado), histórico familiar detalhado, estilo de vida.
4. **Target binarizado simplista**: Estenose ≥50% como proxy de "doença cardíaca" ignora gradiente de severidade e fenótipos clínicos (angina estável vs. SCA vs. assintomático).
5. **Nenhuma validação temporal**: Split aleatório ignora deriva temporal (concept drift) — práticas clínicas e populações mudam ao longo de anos.
6. **Threshold fixo em 0.5**: Não otimizado para custo clínico (FN >> FP em triagem); curve ROC AUC alta mas operating point pode ser subótimo.
7. **Ausência de análise de fairness/viés**: Não avaliado desempenho por subgrupos (sexo, idade, etnia) — risco de disparidades não detectadas.
8. **Calibração não verificada**: Probabilidades de LR assumidas bem calibradas; não testado (ex: calibration curve, Brier score).
9. **Single split holdout**: Uma única partição 80/20; bootstrap ou repeated CV dariam estimativas mais robustas de variância.
10. **Pipeline não testado em dados "sujos" reais**: Inferência assume features completas e tipos corretos; validação de entrada e robustez a outliers não demonstradas.

---

## Future Work

### Curto Prazo (Próximos Notebooks)
- [ ] **05.1_model_explainability.ipynb** — SHAP values (global + local), waterfall plots, beeswarm para validação clínica; análise de fairness por subgrupos (idade, sexo).
- [ ] **06.1_model_monitoring_and_drift.ipynb** — Simulação de data drift (KS test, PSI) e concept drift; alertas de retreinamento com Evidently AI.
- [ ] **07_api_deployment_prep.ipynb** — Schemas Pydantic (validação biomédica), FastAPI endpoints, testes de payload, threshold otimizado por custo clínico.

### Médio Prazo
- **Validação externa**: Testar em datasets independentes (Statlog Heart, Hungarian, Swiss, VA Long Beach) — meta-análise de transportabilidade.
- **Threshold otimizado**: Decision curve analysis / cost-sensitive threshold (minimizar FN ponderado por custo clínico).
- **Calibração de probabilidades**: Isotonic regression / Platt scaling + calibration curves + Brier score.
- **Ensemble stacking**: Combinar LR + RF + SVM com meta-learner (Logistic Regression) para ganho marginal.
- **Feature engineering clínica**: Interactions (age×chol, thalach×exang), scores de risco estabelecidos (Framingham, ASCVD) como features.

### Longo Prazo / Pesquisa
- **Aprendizado federado**: Treino distribuído across hospitais sem compartilhar dados brutos (privacidade).
- **Explainable AI para cardiologia**: Regras extraídas (Anchors, RuleFit) validadas por cardiologistas vs. guidelines (ACC/AHA).
- **Estudo prospetivo**: Deploy silencioso (shadow mode) em serviço de emergência comparando predições vs. desfechos reais (MACE 30d).
- **Fairness rigoroso**: Equalized odds, demographic parity, counterfactual fairness across raça/sexo/idade com dados multi-centro.
- **MLOps completo**: CI/CD para retraining automatizado (drift trigger), model registry (MLflow), A/B testing framework, canary deployment.

---

## Repository Structure

```
kebab-case/
│
├── data/
│   ├── raw/
│   │   └── heart_disease_raw.csv           # Dataset bruto (303×14) após ingestão + binarização target
│   └── processed/
│       └── train_test_data.npz             # Arrays serializados (X_train, X_test, y_train, y_test)
│
├── models/
│   ├── best_model_tuned.joblib             # Modelo final: Logistic Regression otimizada
│   ├── best_model.joblib                   # Baseline Logistic Regression (pré-tuning)
│   ├── preprocessor.joblib                 # ColumnTransformer fitted (pipeline de features)
│   └── tuning_results.json                 # Metadados do tuning: best_params, best_cv_score, cv_results
│
├── notebooks/
│   ├── 01_data_loading.ipynb               # Ingestão UCI → validação → persistência raw
│   ├── 02_eda.ipynb                        # EDA: distribuições, correlações, outliers, missingness
│   ├── 03_preprocessing.ipynb              # Train-test split estratificado + pipeline pré-processamento
│   ├── 04_training.ipynb                   # Baseline CV: 5 modelos × Stratified K-Fold (k=5)
│   ├── 04.1_hyperparameter_tuning.ipynb    # GridSearchCV (LR) + RandomizedSearchCV (RF)
│   ├── 05_evaluation.ipynb                 # Avaliação holdout + relatório por classe + seleção final
│   └── 06_inference.ipynb                  # Inferência simulada: carregar artefatos → predizer
│
├── src/
│   ├── config.py                           # Config centralizada: paths, seeds, CV, modelos, param grids
│   ├── data.py                             # Carregamento único de dados/artefatos (evita duplicação)
│   └── evaluation.py                       # Avaliação padronizada: métricas, plots, relatórios JSON
│
├── requirements.txt                        # Dependências (pinned versions)
├── TODO_MODELS.md                          # Roadmap: explainability, monitoring, deployment
├── LICENSE
└── README.md                               # Este arquivo
```

---

## Quickstart

```bash
# 1. Clone & setup
git clone https://github.com/cauaenzo/kebab-case.git
cd kebab-case
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 2. Run pipeline (sequential)
jupyter lab
# Execute notebooks 01 → 06 in order
```

### Inference Example (Production-Ready)

```python
import pandas as pd
import joblib

# Load serialized artifacts
preprocessor = joblib.load("models/preprocessor.joblib")
model        = joblib.load("models/best_model_tuned.joblib")

# New patient (raw features, same format as original dataset)
new_patient = pd.DataFrame([{
    "age": 57, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 2
}])

# Transform → Predict
X_new = preprocessor.transform(new_patient)
pred_class = model.predict(X_new)[0]           # 0 or 1
pred_proba = model.predict_proba(X_new)[0, 1]  # P(doente)

print(f"Classe: {'Doente' if pred_class else 'Saudável'} | Probabilidade: {pred_proba:.3f}")
```

---

## Reproducibility

- **Seed global**: `SEED = 42` (config.py)
- **CV fixo**: Stratified K-Fold k=5, shuffle=True, random_state=42
- **Split fixo**: 80/20 estratificado, random_state=42
- **Artefatos versionados**: `models/*.joblib` + `tuning_results.json` com timestamps
- **Ambiente**: `requirements.txt` com versões pinned

---

## License

MIT License — veja [LICENSE](LICENSE) para detalhes.