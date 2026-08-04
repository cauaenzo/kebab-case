# Heart Disease Classification — Cleveland Dataset

Projeto de Machine Learning para classificação binária de doenças cardíacas com base no dataset **Cleveland Heart Disease** (UCI Machine Learning Repository).

O objetivo é treinar um modelo capaz de identificar, a partir de variáveis clínicas e demográficas de um paciente, a presença ou ausência de doença cardíaca. O pipeline cobre desde a ingestão dos dados brutos até a inferência em produção, com todos os artefatos serializados e prontos para integração externa.

---

## Estrutura do Repositório

```text
kebab-case/
│
├── data/
│   ├── raw/
│   │   └── heart_disease_raw.csv       # Dataset bruto após ingestão e binarização do target
│   └── processed/
│       └── train_test_data.npz         # Matrizes pré-processadas (X_train, X_test, y_train, y_test)
│
├── models/
│   ├── best_model.joblib               # Modelo final serializado (Logistic Regression)
│   └── preprocessor.joblib            # Pipeline de pré-processamento serializado
│
├── notebooks/
│   ├── 01_data_loading.ipynb          # Ingestão, validação e persistência dos dados brutos
│   ├── 02_eda.ipynb                   # Análise Exploratória de Dados (EDA)
│   ├── 03_preprocessing.ipynb         # Pré-processamento e construção dos pipelines
│   ├── 04_training.ipynb              # Treinamento e comparação de modelos via Cross-Validation
│   ├── 05_evaluation.ipynb            # Avaliação detalhada e seleção do modelo final
│   └── 06_inference.ipynb             # Inferência em novos dados (simulação de produção)
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Fluxo do Pipeline

```mermaid
flowchart TD
    A[01 · Ingestão\nUCI Cleveland Data URL] --> B[02 · EDA\nDistribuições, Correlações, Outliers]
    B --> C[03 · Pré-processamento\nTrain-Test Split Estratificado\nImputation · StandardScaler · OneHotEncoder\nColumnTransformer Pipeline]
    C --> D[04 · Treinamento\nStratified K-Fold CV k=5\nLogistic Regression · Random Forest\nDecision Tree · KNN · SVM]
    D --> E[05 · Avaliação\nConjunto de Teste\nAccuracy · Precision · Recall · F1 · ROC AUC]
    E --> F[Exportação de Artefatos\nbest_model.joblib\npreprocessor.joblib]
    F --> G[06 · Inferência\nCarregamento dos artefatos\nTransformação · Predição · Probabilidade]
```

---

## Resultados e Métricas

### Validação Cruzada — Comparação de Modelos (Notebook 04)

Estratégia: **Stratified K-Fold (k=5)** sobre o conjunto de treino (242 amostras, 28 features após encoding).

| Modelo               | Acurácia Média (CV) | Desvio Padrão |
|----------------------|:-------------------:|:-------------:|
| Logistic Regression  | **0.8471**          | ±0.0100       |
| KNN                  | 0.8264              | ±0.0254       |
| SVM                  | 0.8180              | ±0.0253       |
| Random Forest        | 0.8098              | ±0.0311       |
| Decision Tree        | 0.6897              | ±0.0575       |

> A Regressão Logística obteve a maior acurácia média com o menor desvio padrão, indicando estabilidade superior entre os folds.

---

### Modelo Final — Regressão Logística no Conjunto de Teste (Notebook 05)

Avaliação sobre o conjunto de teste isolado (61 amostras).

| Classe        | Precision | Recall | F1-Score | Support |
|---------------|:---------:|:------:|:--------:|:-------:|
| Saudável (0)  | 0.93      | 0.85   | 0.89     | 33      |
| Doente (1)    | 0.84      | 0.93   | 0.88     | 28      |
| **Macro Avg** | **0.89**  | **0.89** | **0.89** | 61    |

| Métrica    | Valor  |
|------------|:------:|
| Acurácia   | 0.89   |
| ROC AUC    | —      |

> No contexto médico, o **Recall da classe Doente (0.93)** é a métrica crítica: representa a capacidade do modelo de identificar corretamente pacientes doentes, minimizando falsos negativos.

---

## Como Executar o Projeto

### Pré-requisitos

- Python 3.10+
- Git

### 1. Clonar o repositório

```bash
git clone https://github.com/cauaenzo/kebab-case.git
cd kebab-case
```

### 2. Criar e ativar o ambiente virtual

```bash
# Criar
python -m venv venv

# Ativar — Linux/macOS
source venv/bin/activate

# Ativar — Windows
venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Iniciar o JupyterLab

```bash
jupyter lab
```

### 5. Executar os notebooks em ordem

Execute os notebooks de `01_data_loading.ipynb` até `06_inference.ipynb` sequencialmente. Cada notebook depende dos artefatos gerados pelo anterior.

---

## Exemplo de Inferência

O código abaixo replica o fluxo do `06_inference.ipynb` e pode ser integrado em qualquer serviço ou API.

```python
import pandas as pd
import joblib

# 1. Carregar os artefatos serializados
preprocessor = joblib.load("models/preprocessor.joblib")
model        = joblib.load("models/best_model.joblib")

# 2. Dados brutos de um novo paciente (mesmo formato do dataset original)
new_patient = pd.DataFrame([{
    "age":      57,
    "sex":       1,    # 1 = masculino
    "cp":        3,    # tipo de dor no peito
    "trestbps": 145,   # pressão arterial em repouso (mm Hg)
    "chol":     233,   # colesterol sérico (mg/dl)
    "fbs":       1,    # glicemia em jejum > 120 mg/dl
    "restecg":   0,    # resultado do ECG em repouso
    "thalach":  150,   # frequência cardíaca máxima
    "exang":     0,    # angina induzida por exercício
    "oldpeak":   2.3,  # depressão do segmento ST
    "slope":     0,    # inclinação do segmento ST
    "ca":        0,    # número de vasos coloridos por fluoroscopia
    "thal":      1     # resultado da cintilografia miocárdica
}])

# 3. Pré-processar e realizar a predição
processed   = preprocessor.transform(new_patient)
prediction  = model.predict(processed)[0]
probability = model.predict_proba(processed)[0][1]

# 4. Exibir resultado
status = "Presença de Doença Cardíaca" if prediction == 1 else "Saudável / Sem Doença Cardíaca"
print(f"Diagnóstico : {status}")
print(f"Probabilidade de doença: {probability * 100:.2f}%")
```

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

Autor: **cauaenzo**
