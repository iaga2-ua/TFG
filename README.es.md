<div align="center">

# 🏎️ F1 Winner Predictor 2026

**Predicción del ganador de cada Gran Premio de Fórmula 1 antes de la carrera, impulsada por XGBoost y TabNet, desplegada en AWS Lambda con dashboards en vivo en Looker Studio.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-F97316?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![AWS Lambda](https://img.shields.io/badge/AWS-Lambda-FF9900?logo=awslambda&logoColor=white)](https://aws.amazon.com/lambda/)
[![Docker](https://img.shields.io/badge/Docker-contenedorizado-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Looker Studio](https://img.shields.io/badge/Looker_Studio-dashboard_en_vivo-4285F4?logo=googlelookerstudio&logoColor=white)](https://datastudio.google.com/reporting/e898a7b0-6663-46d0-853b-8c2083fab69c)

[**📊 Dashboard en vivo**](https://datastudio.google.com/reporting/e898a7b0-6663-46d0-853b-8c2083fab69c)

</div>

---

## 📈 Resultados

| Modelo | Precisión global | Test en vivo 2026 (R01–R04) | Despliegue |
|:---|:---:|:---:|:---|
| **XGBoost** | **95,9%** (71 / 74 carreras) | **4 / 4 — 100%** | AWS Lambda |
| TabNet | 70,3% (52 / 74 carreras) | 1 / 4 — 25% | Local / Docker |

> Las predicciones se basan únicamente en los datos de la clasificación del sábado. Los modelos se entrenan con datos de 2023–2025 y se evalúan sobre las carreras de 2026 sin reentrenamiento.

---

## 🏗️ Arquitectura

```text
FastF1 API
    │
    ▼
data_collection.py  ──►  feature_engineering.py
                                   │
          ┌────────────────────────┤
          │                        │
    XGBoost (train.py)      TabNet (train.py)
          │                        │
          ▼                        ▼
    S3 (models/)          models/ (local)
          │
          ▼
    AWS Lambda                predict.py (local)
    (predict_lambda.py)            │
          │                        │
          └──────────┬─────────────┘
                     ▼
            S3: predictions/history.csv
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Google Sheets           Athena
    (Looker Studio       (disponible para
      fuente activa)     consultas SQL ad-hoc)
```

---

## 📁 Estructura del proyecto

```text
f1-winner-predictor/
├── config.py                  # Configuración central (paths, features, parámetros)
├── train.py                   # Entrenamiento XGBoost + TabNet con Optuna
├── predict.py                 # Inferencia TabNet local + registro de resultados
├── predict_proba_all.py       # Tabla completa de probabilidades (XGBoost + TabNet, sin escrituras a S3)
├── requirements.txt           # Dependencias del entorno local/Docker
├── requirements-lambda.txt    # Dependencias del contenedor Lambda
├── Dockerfile                 # Imagen para entrenamiento e inferencia local
├── Dockerfile.lambda          # Imagen para AWS Lambda (ECR)
├── Dockerfile.layer           # Capa de dependencias Lambda
├── src/
│   ├── data_collection.py     # Descarga datos de FastF1 (clasificación, carrera, meteorología)
│   ├── feature_engineering.py # Construye las features para XGBoost y TabNet
│   ├── aws_utils.py           # Sincronización con S3 y Google Sheets
│   ├── predict_lambda.py      # Handler de AWS Lambda (inferencia XGBoost)
│   └── circuit_metadata.py    # Tabla estática de metadatos de circuitos
└── models/                    # Artefactos locales (en .gitignore, gestionados vía S3)
    ├── xgboost_f1_winner.pkl
    ├── label_encoders.pkl
    ├── tabnet_model.zip
    └── scaler.pkl
```

---

## 🔢 Features (29 variables)

| Categoría | Feature | Descripción |
|:---|:---|:---|
| **Clasificación** | `grid_position` | Posición en parrilla (sábado, antes de penalizaciones) |
| **Clasificación** | `quali_gap_to_pole_s` | Diferencia con la pole en segundos — `min(Q1, Q2, Q3)` entre todas las sesiones |
| **Entrenos** | `fp2_long_run_pace_gap_s` | Diferencia con el mejor ritmo de carrera en FP2 |
| **Entrenos** | `fp3_gap_to_best_s` | Diferencia con el tiempo más rápido en FP3 |
| **Sprint** | `sprint_race_position` | Posición en la Carrera Sprint (0 = fin de semana estándar) |
| **Meteorología** | `rain_probability` | Fracción de la clasificación con lluvia (0–1) |
| **Meteorología** | `is_wet_qualifying` | Binario: clasificación en mojado |
| **Meteorología** | `track_temp_c`, `humidity_pct`, `wind_speed_ms` | Condiciones durante la clasificación |
| **Campeonato** | `driver_champ_pos`, `driver_champ_points` | Posición y puntos del piloto antes de la carrera |
| **Campeonato** | `constructor_champ_pos`, `constructor_champ_points` | Posición y puntos del constructor |
| **Historial piloto** | `driver_avg_finish_l3` | Posición media de llegada en las últimas 3 carreras |
| **Historial piloto** | `driver_win_rate_l5` | Tasa de victorias en las últimas 5 carreras |
| **Historial piloto** | `driver_wet_win_rate` | Tasa de victorias en carreras mojadas (últimas 10) |
| **Historial piloto** | `driver_current_season_win_rate` | Tasa de victorias en la temporada actual |
| **Historial piloto** | `driver_races_since_last_win` | Carreras desde la última victoria, limitado a 50 |
| **Historial circuito** | `driver_best_finish_circuit`, `driver_avg_finish_circuit` | Mejor y media de resultados en este circuito en los últimos 3 años |
| **Circuito (dinámico)** | `track_length_km`, `corner_count` | Longitud del trazado y número de curvas vía FastF1 |
| **Circuito (estático)** | `overtake_difficulty`, `drs_zones`, `avg_safety_car_prob` | Dificultad de adelantamiento, zonas DRS y probabilidad histórica de safety car |
| **Contexto** | `race_number` | Número de ronda dentro de la temporada |
| **Codificados** | `circuit_encoded`, `constructor_encoded` | Circuito y constructor codificados con label encoding |

---

## 🤖 Modelos

### XGBoost — despliegue en la nube vía AWS Lambda

- Clasificador binario (`is_winner = 1` para el ganador de la carrera)
- `scale_pos_weight = 5`: corrección parcial del desequilibrio de clases (el valor completo de 21 concentra demasiado las probabilidades)
- `base_score = 0.045` (= 1/22): prior natural de victoria para evitar estimaciones infladas en pilotos con pocos datos
- Hiperparámetros ajustados con **Optuna** (ROC-AUC, validación cruzada estratificada de 5 pliegues)
- Entrenado con **pesos de recencia** `{2023:1, 2024:1, 2025:2, 2026:2}` vía `sample_weight`
- Artefactos en S3: `xgboost_f1_winner.pkl`, `label_encoders.pkl`, `xgb_temperature.pkl`

### TabNet — inferencia local

- Red neuronal tabular con mecanismo de atención secuencial (`pytorch-tabnet`)
- Pre-estandarizado con `StandardScaler`
- Hiperparámetros ajustados con **Optuna** (ROC-AUC, 20% de validación)
- Entrenado con **sobremuestreo por recencia** `{2023:1, 2024:1, 2025:2, 2026:2}` mediante duplicación de filas
- Artefactos: `tabnet_model.zip`, `scaler.pkl`, `tabnet_temperature.pkl`

> **¿Por qué dos modelos?** TabNet requiere PyTorch (~1 GB de dependencias), incompatible con las restricciones de tamaño de AWS Lambda. XGBoost es ligero (~50 MB) y se despliega sin problemas. Ejecutar ambos modelos permite una comparación arquitectónica directa.

### Calibración de probabilidades — Temperature Scaling

Ambos modelos se calibran a posteriori con **Temperature Scaling** ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)).

**Por qué es necesaria la calibración:**
1. Sin calibración, el piloto en pole puede recibir ~90–99% de probabilidad de victoria, muy por encima de la tasa real (~40–45%)
2. Los 22 clasificadores binarios independientes de XGBoost no suman 100% de forma natural

**Cómo funciona:** cada modelo se entrena con el 80% de los datos. Sobre el 20% restante, se minimiza la NLL con L-BFGS para encontrar la temperatura óptima $T$:

$$p_{\text{cal}} = \frac{1}{1 + e^{-z/T}}, \qquad p_{\text{norm}} = \frac{p_{\text{cal}}}{\displaystyle\sum_i p_{\text{cal},i}}$$

donde $z = \log\frac{p}{1-p}$ son los logits extraídos de la salida bruta del modelo. $T > 1$ aplana las probabilidades y reduce el exceso de confianza.

---

## ☁️ Infraestructura AWS

| Recurso | Detalle |
|:---|:---|
| **Región** | `eu-west-1` |
| **Bucket S3** | `f1-winner-predictor-2026` |
| **Función Lambda** | `f1-winner-predictor` (imagen ECR) |
| **ECR** | `606756239522.dkr.ecr.eu-west-1.amazonaws.com/f1-winner-predictor:latest` |
| **Athena** | BD: `f1_predictions` / Tablas: `race_predictions`, `feature_importance` |

### Estructura del bucket S3

```text
f1-winner-predictor-2026/
├── predictions/
│   └── history.csv                   # Una fila por carrera (XGBoost + TabNet fusionados)
├── models/
│   ├── xgboost_f1_winner.pkl
│   ├── label_encoders.pkl
│   ├── xgb_temperature.pkl
│   └── tabnet_temperature.pkl
├── metrics/
│   ├── feature_importance.csv
│   └── historical_performance.csv
└── data/
    └── race_results_raw.csv
```

---

## 📊 Google Sheets y Looker Studio

Las predicciones y métricas se sincronizan automáticamente con Google Sheets (enlazado como fuente de datos de Looker Studio):

| Pestaña | Contenido | Actualización |
|:---|:---|:---|
| **Sheet1** | Historial completo de predicciones con resultados 2026 | Sábado (predicción) y lunes (registro del resultado) |
| **feature_importance** | Importancia de cada feature para ambos modelos | Al reentrenar con `--upload-s3` |
| **model_accuracy** | Precisión acumulada, Brier Score y MAE posicional | Lunes (tras cada carrera) |

**[Ver dashboard en vivo →](https://datastudio.google.com/reporting/e898a7b0-6663-46d0-853b-8c2083fab69c)**

---

## 📐 Métricas de evaluación

Se calculan tres métricas complementarias de forma acumulativa tras cada carrera:

### Precisión
$$\text{Precisión} = \frac{\text{predicciones correctas}}{\text{carreras completadas}}$$

### Brier Score (menor es mejor, 0 = perfecto)
$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$$

$p_i$ es la probabilidad asignada al piloto predicho e $y_i \in \{0,1\}$ indica si ganó realmente. Penaliza mucho más las predicciones erróneas con alta confianza que las cautelosas.

### MAE posicional (menor es mejor, 0 = ganador predicho correctamente)
$$\text{MAE}_{\text{pos}} = \frac{1}{N}\sum_{i=1}^{N}|f_i - 1|$$

$f_i$ es la posición de llegada del piloto que el modelo predijo como ganador. Cero si el modelo acertó, cuatro si el elegido terminó en P5.

---

## 🚀 Flujo de trabajo

### Viernes (opcional) — reentrenar modelos

```powershell
# Reentrenamiento completo con ajuste Optuna y subida a S3
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data --optimize

# Reentrenamiento rápido sin búsqueda de hiperparámetros
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data

# Modelo individual
docker compose run --rm trainer python train.py --model xgboost --upload-s3
docker compose run --rm trainer python train.py --model tabnet
```

### Sábado — predicción tras la clasificación

```powershell
# XGBoost vía Lambda
aws lambda invoke --function-name f1-winner-predictor `
  --payload '{"year":2026,"round":5}' response.json --region eu-west-1

# TabNet vía Docker
docker compose run --rm trainer python predict.py --round 5 --year 2026
```

### Sábado — tabla completa de probabilidades (sin escrituras a S3)

```powershell
docker compose run --rm proba --round 5
```

```text
+==================================================+
|      2026  |  Round 3  |  JAPANESE GRAND PRIX    |
+==================================================+
  #   DRIVER      XGBoost     TabNet
  ----------------------------------------------------
  1   ANT           60.0%      50.0% <--
  2   RUS           21.1%      47.6%
  3   PIA            4.2%       0.7%
  4   HAM            2.3%       0.2%
  5   VER            2.1%       0.0%
```

### Lunes — registrar el resultado real

```powershell
docker compose run --rm trainer python predict.py --round 5 --year 2026 --record-result
```

Descarga el resultado de la carrera desde FastF1, escribe `actual_winner` y las columnas de precisión en S3 y actualiza las tres pestañas de Google Sheets.

---

## ⚙️ Script de despliegue (`deploy.ps1`)

| Modo | Qué hace |
|:---|:---|
| `models` | Reentrena XGBoost + TabNet y sube los artefactos a S3 |
| `lambda` | Reconstruye la imagen Docker, la sube a ECR y actualiza Lambda |
| `predict` | Invoca Lambda + ejecuta TabNet local y guarda el snapshot de probabilidades |
| `all` | Ejecuta `models`, `lambda` y `predict` en secuencia |

```powershell
.\deploy.ps1 -Mode lambda                    # Redesplegar solo Lambda
.\deploy.ps1 -Mode models                    # Reentrenar y subir a S3
.\deploy.ps1 -Mode all -Refresh              # Pipeline completo con datos frescos
.\deploy.ps1 -Mode all -Refresh -Optimize    # Pipeline completo con ajuste Optuna
```

---

## 🏁 Fines de semana con Sprint

Los fines de semana Sprint tienen un calendario de sesiones diferente que requiere dos ajustes:

| Problema | Causa | Solución |
|:---|:---|:---|
| FP2/FP3 no disponibles | Las sesiones no existen en fines de semana Sprint | Ritmo de FP1 usado como sustituto en ambas columnas |
| Puntos subestimados | Puntos de la Carrera Sprint omitidos en `race_results_raw.csv` | Puntos de la sesión Sprint (`"S"`) añadidos a los de la carrera principal en la ingesta |

| Feature | Sin ajuste | Con ajuste |
|:---|:---:|:---:|
| `fp2_long_run_pace_gap_s` | `0` para todos los pilotos | Diferencia real desde FP1 |
| `fp3_gap_to_best_s` | `0` para todos los pilotos | Diferencia real desde FP1 |
| `driver_champ_points` | Subestimado | Exacto (carrera principal + sprint) |
| `constructor_champ_points` | Subestimado | Exacto (carrera principal + sprint) |

---

## 🔧 Configuración inicial

### 1. Variables de entorno

Crea el archivo `.env` en la raíz del proyecto:

```env
AWS_PROFILE=f1-developer
AWS_REGION=eu-west-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 2. Credenciales de Google Sheets

Descarga el JSON de la cuenta de servicio `f1-sheets-sync@f1-tfg.iam.gserviceaccount.com` y colócalo en:

```text
f1-winner-predictor/.google_credentials.json
```

> Este archivo está en `.gitignore` y no debe subirse nunca al repositorio.

### 3. Construir la imagen Docker

```powershell
cd "TFG"
docker build -t f1-tabnet:latest .
```

