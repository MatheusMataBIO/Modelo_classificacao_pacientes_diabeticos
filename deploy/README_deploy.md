---
title: Triagem de Diabetes Mellitus
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: false
license: mit
---

# 🩺 Triagem de Diabetes Mellitus — CDC BRFSS 2015

Modelo preditivo para triagem de diabetes mellitus treinado com **229.474 pacientes reais**
do CDC americano (Behavioral Risk Factor Surveillance System 2015).

## Performance

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,8192 |
| Recall | 0,8168 |
| Precisão | 0,3059 |
| Threshold | 0,47 |
| Algoritmo | LightGBM |

> ⚠️ Este modelo é uma ferramenta de triagem e **não substitui avaliação médica profissional**.
