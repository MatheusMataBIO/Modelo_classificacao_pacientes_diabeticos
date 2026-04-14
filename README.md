# 🚀 Previsão de Diabetes Mellitus com Machine Learning

> Modelo preditivo de triagem para diabetes mellitus treinado com **229.474 pacientes reais** do CDC americano — sem uso de exames laboratoriais.

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-green?style=flat-square)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Deploy-Streamlit-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-Spaces-yellow?style=flat-square)](https://huggingface.co/spaces)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?style=flat-square)](https://mlflow.org)

---

## 🔗 Demo ao Vivo

**👉 [Acesse a aplicação no Hugging Face Spaces](https://matheusmata-modelo-preditivo-diabetes.hf.space)**

Preencha os dados do paciente e receba em segundos:
- Diagnóstico (COM / SEM diabetes)
- Probabilidade individual (%)
- Gauge visual de nível de risco
- Explicação dos fatores determinantes (SHAP)
- Comparação com a população de 45.895 pacientes

---

## 📋 Contexto

O **diabetes mellitus tipo 2** é uma das doenças crônicas mais prevalentes do mundo. Nos Estados Unidos, cerca de **38 milhões de pessoas** vivem com a doença — e aproximadamente **1 em cada 5 não sabe que a tem**. O diagnóstico tardio agrava complicações como doenças cardiovasculares, insuficiência renal e neuropatias.

Este projeto utiliza o dataset **CDC BRFSS 2015** (Behavioral Risk Factor Surveillance System), a maior pesquisa telefônica de saúde dos EUA conduzida anualmente pelo CDC. São 253.680 adultos americanos reais respondendo perguntas sobre comportamentos de saúde, condições crônicas e uso de serviços médicos — **sem nenhum exame laboratorial**.

### Por que esse dataset é especial?

Todas as variáveis são baseadas em **autorrelato**. Isso significa que o modelo pode ser aplicado em contextos de triagem de baixo custo — aplicativos de saúde, UBS, campanhas de saúde pública — onde glicemia e HbA1c não estão disponíveis.

---

## ❓ Problema

> **É possível prever se um indivíduo tem diabetes usando apenas variáveis comportamentais, sociodemográficas e de autoavaliação de saúde — sem depender de exames laboratoriais?**

### Por que é difícil?

- **Desbalanceamento severo:** 86,1% sem diabetes vs 13,9% com diabetes (razão 6,2:1)
- **Sem exames laboratoriais:** nenhuma variável como glicemia, HbA1c ou insulina
- **Variáveis binárias dominantes:** 14 das 21 features são 0/1, o que inviabiliza técnicas como SMOTE
- **Custo assimétrico dos erros:** um falso negativo (diabético não identificado) é muito mais grave que um falso positivo

### Métricas prioritárias

Por essas razões, **acurácia não é a métrica adequada**. Um modelo que chutasse sempre 0 teria 86% de acurácia sem aprender nada. As métricas priorizadas foram:

| Prioridade | Métrica | Justificativa |
|---|---|---|
| 1ª | **Recall** | Minimiza falsos negativos — não deixar diabéticos sem identificar |
| 2ª | **ROC-AUC** | Mede capacidade discriminativa independente do threshold |
| 3ª | **F1-Score** | Equilíbrio como regularizador secundário |

---

## 🗂️ Estrutura do Projeto

```
diabetes-mellitus-prediction/
│
├── 📓 notebooks/
│   ├── 01_eda_diabetes_brfss.ipynb           # Análise Exploratória de Dados
│   ├── 02_preprocessamento_diabetes.ipynb    # Pré-processamento e Feature Engineering
│   ├── 03_modelagem_diabetes.ipynb           # Modelagem, Optuna e SHAP
│   ├── 04_mlflow_diabetes.ipynb              # Rastreamento com MLflow
│   └── 05_deploy_huggingface.ipynb           # Preparação para Deploy
│
├── 🚀 deploy/
│   ├── app.py                                # Interface Streamlit (Hugging Face)
│   ├── requirements.txt                      # Dependências do deploy
│   └── README.md                             # Configuração do HF Space
│
├── 📄 relatorio_tecnico_diabetes.pdf         # Relatório técnico completo
│
└── README.md
```

> **Nota:** Os artefatos do modelo (`.pkl`, `.py`, `.json`) são gerados localmente ao executar os notebooks e não estão versionados no repositório.

---

## 🔬 Metodologia

### Pipeline Completo

```
Dados Brutos (253.680)
       ↓
  Remoção de Duplicatas (-24.206)
       ↓
  Winsorização do IMC (p99)
       ↓
  Feature Engineering (+6 features)
       ↓
  Split 80/20 Estratificado
       ↓
  Pipeline: OutlierClipper → RobustScaler
       ↓
  class_weight='balanced' (sem SMOTE)
       ↓
  Baseline → Comparação → Campeão
       ↓
  Optuna (80 trials, métrica composta)
       ↓
  Ajuste de Threshold (max Recall)
       ↓
  SHAP → MLflow → Deploy
```

### Features Criadas (Feature Engineering)

| Feature | Descrição | Motivação |
|---|---|---|
| `idade_x_saude_geral` | Interação faixa etária × saúde geral | Heatmap EDA mostrou efeito combinado forte |
| `n_fatores_risco` | Contagem de 5 fatores acumulados | Taxa DM: 2,5% (0 fatores) → 60% (5 fatores) |
| `imc_categoria` | Categorização clínica OMS (0–4) | Inflexão relevante no limiar 30 (obesidade I) |
| `score_saudavel` | Soma de comportamentos protetores | Agrega fatores protetores identificados na EDA |
| `score_socioeconomico` | Renda + escolaridade | Gradiente socioeconômico consistente |
| `alto_risco_combinado` | Flag: idoso + saúde ruim | 37,1% vs 12,6% de prevalência de DM |

### Por que `class_weight` em vez de SMOTE?

O SMOTE gera amostras sintéticas interpolando entre vizinhos no espaço de features. Para variáveis binárias (14 das 21 features), isso produziria valores como 0,3 ou 0,7 — sem significado clínico. Com 229 mil registros, o problema não é falta de dados, mas proporção entre classes. O `class_weight='balanced'` penaliza erros na classe minoritária sem criar dados sintéticos.

---

## 📊 Resultados

### Comparação de Modelos

| Modelo | ROC-AUC | Recall | F1 | Aprovado? |
|---|---|---|---|---|
| ⭐ Baseline (Log. Reg.) | 0,8125 | 0,7669 | 0,4493 | Referência |
| Decision Tree | 0,8080 | 0,7789 | 0,4395 | ❌ AUC < baseline |
| Random Forest | 0,8163 | 0,7733 | 0,4515 | ✅ |
| Gradient Boosting | 0,8195 | **0,1670** | 0,2603 | ❌ **Recall crítico** |
| XGBoost | 0,8188 | 0,7847 | 0,4571 | ✅ |
| **LightGBM** ⬅ campeão | **0,8190** | 0,7843 | 0,4550 | ✅ |

> ⚠️ O Gradient Boosting foi **descartado** apesar do maior ROC-AUC por apresentar Recall de apenas 16,7% — clinicamente inaceitável. Isso ilustra por que AUC isolado não é suficiente para problemas médicos.

### Otimização com Optuna

A função objetivo usou uma **métrica composta** para evitar que o tuning sacrificasse o Recall:

```python
# Descarta trial se Recall < baseline
if recall_cv < recall_minimo:
    return 0.0

# Métrica composta: prioriza AUC sem ignorar Recall e F1
return 0.5 * auc_cv + 0.3 * recall_cv + 0.2 * f1_cv
```

**80 trials** com `TPESampler` (busca bayesiana) | **75 válidos** | **5 descartados** por Recall insuficiente

### Ajuste de Threshold

O threshold padrão de 0,5 não é adequado para problemas clínicos. O critério adotado foi:

> **Máximo Recall com Precisão ≥ 2× a prevalência real (~30%)**

| Threshold | Recall | Precisão | F1 | Observação |
|---|---|---|---|---|
| 0,50 (padrão) | 0,7930 | 0,3168 | 0,4527 | Padrão |
| 0,65 (por F1) | 0,5945 | 0,4003 | 0,4784 | Sacrifica Recall |
| **0,47 (ótimo)** | **0,8168** | 0,3059 | 0,4451 | **Escolhido** |

### Resultado Final

| Métrica | Baseline (LR) | LightGBM Final | Ganho |
|---|---|---|---|
| ROC-AUC | 0,8125 | **0,8192** | +0,0067 |
| **Recall** | 0,7669 | **0,8168** | **+0,0499** |
| Precisão | 0,3177 | 0,3059 | -0,0118 |
| F1-Score | 0,4493 | 0,4451 | -0,0042 |
| Threshold | 0,50 | **0,47** | Otimizado |

### Matriz de Confusão (threshold = 0,47)

```
                 Predito: Sem DM    Predito: Com DM
Real: Sem DM  |    25.866 (TN)   |   13.010 (FP)  |
Real: Com DM  |     1.286 (FN)   |    5.733 (TP)  |
```

- **81,7%** dos pacientes diabéticos identificados corretamente
- **95%** de precisão ao classificar como "sem diabetes" — seguro para descartar diagnóstico

---

## 🧬 Interpretabilidade (SHAP)

As features mais importantes identificadas pelo SHAP:

```
1. idade_x_saude_geral    ████████████████  0.312  (criada no feature eng.)
2. n_fatores_risco        ██████████████    0.287  (criada no feature eng.)
3. imc                    ██████████        0.198
4. saude_geral            █████████         0.187
5. pressao_alta           ███████           0.143
6. faixa_etaria           ██████            0.121
7. checou_colesterol      █████             0.098
8. dias_saude_mental_ruim ████              0.087
```

> As duas features criadas no feature engineering lideram a importância global, validando as decisões tomadas com base na EDA.

---

## 💡 Impacto

### Clínico

O modelo identifica **81,7% dos pacientes diabéticos** usando apenas um questionário de autorrelato — sem exames laboratoriais. Quando classifica alguém como "sem diabetes", está certo em **95% dos casos**, tornando-o seguro para triagem em larga escala.

### Operacional

Um sistema de triagem baseado neste modelo poderia ser aplicado em:
- **Aplicativos de saúde** — triagem inicial antes de consulta médica
- **Unidades Básicas de Saúde** — pré-seleção de pacientes para exames confirmatórios
- **Campanhas de saúde pública** — identificação de populações de alto risco
- **Planos de saúde** — programas preventivos direcionados

### Econômico

Considerando os 45.895 pacientes do conjunto de teste, o modelo reduziria a necessidade de exames laboratoriais para triagem de **100%** para apenas os **40,3%** classificados como positivos — mantendo 81,7% de sensibilidade diagnóstica.

---

## 🧪 Principais Decisões Técnicas

| Decisão | Alternativa Descartada | Justificativa |
|---|---|---|
| `class_weight='balanced'` | SMOTE | 14 features binárias — SMOTE gera valores sem sentido clínico |
| Threshold por max Recall | Threshold por F1 | F1 otimizaria para 0,65, reduzindo Recall de 79% para 59% |
| Métrica composta no Optuna | Apenas AUC | AUC sozinho selecionaria Gradient Boosting com Recall de 16,7% |
| Filtro de Recall ≥ baseline | Apenas AUC para ranking | Garante que o campeão seja clinicamente viável |
| LightGBM raso (depth=3) | Modelos mais profundos | Regularização evita overfitting + baixa variância no CV |

---

## 🛠️ Stack Tecnológico

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.10 |
| Modelagem | scikit-learn, LightGBM, XGBoost |
| Otimização | Optuna (TPESampler, 80 trials) |
| Interpretabilidade | SHAP (TreeExplainer) |
| Rastreamento | MLflow (87 runs registradas) |
| Interface | Streamlit 1.41.0 |
| Deploy | Hugging Face Spaces (gratuito) |
| Ambiente | Google Colab |

---

## 🚀 Como Executar Localmente

### Pré-requisitos

```bash
git clone https://github.com/SEU_USUARIO/diabetes-mellitus-prediction
cd diabetes-mellitus-prediction
pip install -r deploy/requirements.txt
```

### Executar os Notebooks

Execute os notebooks na ordem numérica no Google Colab ou Jupyter:

```
01_eda_diabetes_brfss.ipynb
02_preprocessamento_diabetes.ipynb
03_modelagem_diabetes.ipynb
04_mlflow_diabetes.ipynb
05_deploy_huggingface.ipynb
```

> Os notebooks fazem download automático do dataset via `kagglehub`. É necessário ter uma conta no Kaggle e o arquivo `kaggle.json` configurado.

---

## 📁 Dataset

**CDC BRFSS 2015** — Behavioral Risk Factor Surveillance System

| Atributo | Valor |
|---|---|
| Registros originais | 253.680 |
| Após limpeza | 229.474 |
| Features originais | 21 |
| Features após engineering | 27 |
| Prevalência de diabetes | 13,9% |
| Desbalanceamento | 6,2:1 |
| Valores nulos | 0 |

- 📥 Kaggle: [alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- 🏛️ Fonte original: [CDC BRFSS](https://www.cdc.gov/brfss)

---

## ✅ Conclusões

1. **É possível** prever diabetes com ROC-AUC de 0,82 usando apenas variáveis de autorrelato — sem exames laboratoriais.

2. **Feature Engineering importa:** a feature `idade_x_saude_geral`, criada com base nos padrões observados na EDA, emergiu como a mais importante do modelo, superando todas as originais.

3. **Threshold importa mais que o algoritmo:** o ajuste de threshold gerou o maior ganho individual de Recall do projeto (+0,05), acima do ganho obtido com Optuna (+0,008).

4. **ROC-AUC sozinho engana:** o Gradient Boosting tinha o maior AUC mas Recall de 16,7% — clinicamente inaceitável. A definição de critérios clínicos de aprovação é essencial em projetos de saúde.

5. **Desbalanceamento não precisa de oversampling:** `class_weight='balanced'` foi mais adequado que SMOTE para este dataset por preservar a integridade das variáveis binárias.

6. **Rastreabilidade é profissionalismo:** o MLflow com 87 runs registradas garante que cada decisão do projeto seja reproduzível e auditável.

---

## ⚠️ Aviso Importante

> Este modelo é uma **ferramenta de triagem** e **não substitui avaliação médica profissional**. Os resultados devem ser interpretados por profissionais de saúde qualificados. Um resultado positivo indica necessidade de exames confirmatórios — não constitui diagnóstico definitivo.

---

<div align="center">

**Desenvolvido com 🩺 e ☕**

*Se este projeto foi útil, considere dar uma ⭐*

</div>
