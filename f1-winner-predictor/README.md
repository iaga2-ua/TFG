# F1 Winner Predictor 2026

Predictor del ganador de cada Gran Premio de Fórmula 1 basado en datos de clasificación (sábado). Combina dos modelos de machine learning — **XGBoost** (desplegado en AWS Lambda) y **TabNet** (inferencia local) — y sincroniza predicciones y métricas con Google Sheets para visualización en Looker Studio.

---

## Arquitectura general

```
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
     — fuente activa)    consultas SQL ad-hoc)
```

---

## Estructura del proyecto

```
f1-winner-predictor/
├── config.py                  # Configuración central (paths, features, parámetros)
├── train.py                   # Entrenamiento XGBoost + TabNet con Optuna
├── predict.py                 # Inferencia TabNet local + registro de resultados
├── requirements.txt           # Dependencias del entorno local/Docker
├── requirements-lambda.txt    # Dependencias del contenedor Lambda
├── Dockerfile                 # Imagen para entrenamiento y predicción local
├── Dockerfile.lambda          # Imagen para AWS Lambda (ECR)
├── Dockerfile.layer           # Layer de dependencias Lambda
├── src/
│   ├── __init__.py
│   ├── data_collection.py     # Descarga datos de FastF1 (quali, carrera, meteorología)
│   ├── feature_engineering.py # Construcción de features para XGBoost y TabNet
│   ├── aws_utils.py           # S3, Google Sheets sync (Athena configurado para SQL ad-hoc)
│   ├── predict_lambda.py      # Handler de AWS Lambda (inferencia XGBoost)
│   └── circuit_metadata.py    # Tabla estática de metadatos de circuitos
└── models/                    # Artefactos locales (gitignored, gestionados vía S3)
    ├── xgboost_f1_winner.pkl
    ├── label_encoders.pkl
    ├── tabnet_model.zip
    └── scaler.pkl
```

---

## Features utilizadas (26 variables)

| Categoría | Feature | Descripción |
|---|---|---|
| **Clasificación** | `grid_position` | Posición en parrilla (sábado, no penalizaciones) |
| **Clasificación** | `quali_gap_to_pole_s` | Gap al pole en segundos. Se calcula como `min(Q1, Q2, Q3)` para capturar la mejor vuelta real de cada piloto independientemente de la sesión en que fue eliminado |
| **Práctica libre** | `fp2_long_run_pace_gap_s` | Gap al mejor ritmo de carrera en FP2 |
| **Práctica libre** | `fp3_gap_to_best_s` | Gap al mejor tiempo en FP3 |
| **Meteorología** | `rain_probability` | Fracción de la sesión con lluvia (0–1) |
| **Meteorología** | `is_wet_qualifying` | Binario: clasificación en mojado |
| **Meteorología** | `track_temp_c`, `humidity_pct`, `wind_speed_ms` | Condiciones durante la clasificación |
| **Campeonato** | `driver_champ_pos`, `driver_champ_points` | Posición y puntos del piloto antes de la carrera |
| **Campeonato** | `constructor_champ_pos`, `constructor_champ_points` | Posición y puntos del constructor |
| **Historial piloto** | `driver_avg_finish_l3` | Media de posición final en las últimas 3 carreras |
| **Historial piloto** | `driver_win_rate_l5` | Tasa de victorias en las últimas 5 carreras |
| **Historial piloto** | `driver_wet_win_rate` | Tasa de victorias en carreras mojadas (últimas 10) |
| **Historial circuito** | `driver_best_finish_circuit`, `driver_avg_finish_circuit` | Mejor y media de resultado en este circuito |
| **Circuito (FastF1)** | `track_length_km`, `corner_count` | Longitud y número de curvas (dinámico) |
| **Circuito (estático)** | `overtake_difficulty`, `drs_zones`, `avg_safety_car_prob` | Dificultad de adelantamiento, zonas DRS, probabilidad histórica de SC |
| **Contexto** | `race_number` | Número de ronda en la temporada |
| **Codificados** | `circuit_encoded`, `constructor_encoded` | Label encoding de circuito y constructor |

---

## Modelos

### XGBoost (nube — AWS Lambda)
- Clasificador binario (`is_winner = 1` para el ganador de cada carrera)
- `scale_pos_weight = 21` para compensar el desbalance (1 ganador vs 21 pilotos)
- Hiperparámetros optimizados con **Optuna** (ROC-AUC, 5-fold CV estratificado)
- Entrenado con datos 2023–2026 (solo carreras ya disputadas)
- Artefactos en S3: `models/xgboost_f1_winner.pkl` + `models/label_encoders.pkl`

### TabNet (local)
- Red neuronal tabular con mecanismo de atención (pytorch-tabnet)
- Mismos datos y features que XGBoost
- Estandarización previa con `StandardScaler`
- Hiperparámetros optimizados con **Optuna** (ROC-AUC, validación 20%)
- Artefactos en local: `models/tabnet_model.zip` + `models/scaler.pkl`

> **Razón del diseño dual**: TabNet requiere PyTorch (~1 GB de dependencias), incompatible con los límites prácticos de AWS Lambda. XGBoost es ligero (~50 MB) y se despliega sin problemas. Correr ambos modelos permite comparar sus predicciones directamente.

---

## Infraestructura AWS

| Recurso | Detalles |
|---|---|
| **Región** | `eu-west-1` |
| **S3 bucket** | `f1-winner-predictor-2026` |
| **Lambda** | `f1-winner-predictor` (imagen ECR) |
| **ECR** | `606756239522.dkr.ecr.eu-west-1.amazonaws.com/f1-winner-predictor:latest` |
| **Athena** | DB: `f1_predictions`, tablas: `race_predictions`, `feature_importance` — configurado para consultas SQL ad-hoc; Looker Studio usa Google Sheets como fuente activa |

### Estructura del bucket S3

```
f1-winner-predictor-2026/
├── predictions/
│   └── history.csv              # Una fila por carrera (XGB + TabNet fusionados)
├── models/
│   ├── xgboost_f1_winner.pkl
│   └── label_encoders.pkl
├── metrics/
│   ├── feature_importance.csv   # Importancia de features por modelo
│   └── historical_performance.csv  # Predicciones vs resultados (datos de entrenamiento)
└── data/
    └── race_results_raw.csv     # Datos históricos de entrenamiento
```

### Fusión XGB + TabNet en history.csv

Lambda escribe `predicted_winner_xgb` y el script local escribe `predicted_winner_tab`. La función `append_to_history_csv()` fusiona ambas escrituras en la **misma fila** (indexed by `year` + `round`) sin sobreescribir columnas ya escritas por el otro modelo.

---

## Google Sheets + Looker Studio

Las predicciones y métricas se sincronizan automáticamente con Google Sheets (ID: `1Jw7wo3bqC2IS9MmfSJe6T2waQyp7LTCMv4al7gwhtPI`):

| Pestaña | Contenido | Cuándo se actualiza |
|---|---|---|
| **Sheet1** | `history.csv` — predicciones 2026 por carrera | Al predecir (sábado) y al registrar resultado (lunes) |
| **feature_importance** | Importancia de cada feature en XGBoost y TabNet | Al reentrenar con `--upload-s3` |
| **model_accuracy** | Accuracy acumulada de ambos modelos carrera a carrera | Al registrar el ganador real (lunes) |

### Gráficos en Looker Studio

| Gráfico | Fuente | Configuración |
|---|---|---|
| **Tabla de predicciones** | Sheet1 | Dimensiones: `event_name`, `predicted_winner_xgb`, `predicted_winner_tab`, `actual_winner`, `xgb_correct`, `tab_correct` / Métricas: `win_prob_xgboost`, `win_prob_tabnet` |
| **Barras de importancia** | `feature_importance` | Dimensión Y: `feature` / Métricas X: `importance_xgboost` + `importance_tabnet` |
| **Línea de accuracy** | `model_accuracy` | Dimensión X: `race_label` / Métricas Y: `xgb_accuracy_cumul` + `tab_accuracy_cumul` |

---

## Configuración inicial (primera vez)

### 1. Variables de entorno

Crea `.env` en la raíz del proyecto (`TFG/`):

```env
AWS_PROFILE=f1-developer
AWS_REGION=eu-west-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 2. Credenciales de Google Sheets

Descarga el JSON de la service account `f1-sheets-sync@f1-tfg.iam.gserviceaccount.com` y colócalo en:
```
f1-winner-predictor/.google_credentials.json
```
Este archivo está en `.gitignore` y nunca debe subirse al repositorio.

### 3. Construir la imagen Docker

```powershell
cd "TFG"
docker build -t f1-tabnet:latest .
```

---

## Flujo semanal de uso

### Viernes (opcional) — Reentrenar modelos

Solo necesario si hay nuevas carreras disputadas o se quieren mejorar los modelos.

```powershell
# Entrenamiento completo con optimización de hiperparámetros y subida a S3:
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data --optimize

# Sin optimización (más rápido, usa hiperparámetros por defecto):
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data

# Solo un modelo:
docker compose run --rm trainer python train.py --model xgboost --upload-s3
docker compose run --rm trainer python train.py --model tabnet
```

Al terminar se actualizan automáticamente:
- XGBoost y encoders en S3 (disponibles para Lambda)
- TabNet y scaler en `models/` local
- Pestaña `feature_importance` de Google Sheets

### Después de la clasificación (sábado) — Predecir

**XGBoost vía Lambda** (invocar manualmente tras la clasificación):

```powershell
aws lambda invoke --function-name f1-winner-predictor `
  --payload '{"year":2026,"round":2}' response.json --region eu-west-1
```

**TabNet vía Docker** (manual, post-clasificación):

```powershell
docker compose run --rm trainer python predict.py --round 2 --year 2026
```

Ambas predicciones se fusionan en una única fila en S3 y se sincronizan con Google Sheets.

### Después de la carrera (lunes) — Registrar resultado real

```powershell
# FastF1 descarga automáticamente el ganador desde la API:
docker compose run --rm trainer python predict.py --round 2 --year 2026 --record-result
```

Esto:
1. Descarga el resultado desde FastF1
2. Escribe `actual_winner`, `xgb_correct` y `tab_correct` en S3
3. Sincroniza Sheet1 con el resultado real
4. Actualiza la pestaña `model_accuracy` con la accuracy acumulada

---

## Rebuilding y despliegue de la Lambda

Obligatorio cuando se modifica `predict_lambda.py`, `aws_utils.py` o `feature_engineering.py`:

```powershell
# Desde f1-winner-predictor/
docker build -f Dockerfile.lambda -t f1-winner-predictor .

aws ecr get-login-password --region eu-west-1 | `
  docker login --username AWS --password-stdin 606756239522.dkr.ecr.eu-west-1.amazonaws.com

docker tag f1-winner-predictor:latest `
  606756239522.dkr.ecr.eu-west-1.amazonaws.com/f1-winner-predictor:latest
docker push 606756239522.dkr.ecr.eu-west-1.amazonaws.com/f1-winner-predictor:latest

aws lambda update-function-code --function-name f1-winner-predictor `
  --image-uri 606756239522.dkr.ecr.eu-west-1.amazonaws.com/f1-winner-predictor:latest `
  --region eu-west-1
aws lambda wait function-updated --function-name f1-winner-predictor --region eu-west-1
```

---

## Seguridad

- Las credenciales AWS van en `.env` (gitignored) o en variables de entorno del sistema
- Las credenciales de Google van en `.google_credentials.json` (gitignored)
- Los artefactos de modelo (`models/*.pkl`, `models/*.zip`) y los datos (`data/`) están en `.gitignore`
- El directorio `package/` (build de Lambda) está en `.gitignore`
- Nunca se hardcodean claves en el código fuente

