# 🏎 F1 Winner Predictor — XGBoost + AWS

Predicts the winner of each 2026 Formula 1 race using XGBoost.  
Inference runs **post-qualifying (Saturday)** once the starting grid is known.  
Predictions are stored in **Amazon S3** and queryable via **Amazon Athena / Looker Studio**.

---

## Architecture

```
FastF1 API ──► predict.py ──► XGBoost model ──► history.csv (S3)
                                                       │
                                               Amazon Athena
                                                       │
                                               Looker Studio
```

---

## Project structure

```
f1-winner-predictor/
├── config.py                  ← Configuración central (rutas, AWS, parámetros)
├── train.py                   ← Entrenamiento dual XGBoost + TabNet
├── predict.py                 ← Inferencia post-clasificación (sábado)
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/                       ← Módulos de soporte (importados por los scripts raíz)
│   ├── __init__.py
│   ├── aws_utils.py           ← Operaciones S3 + Athena (boto3)
│   ├── circuit_metadata.py    ← Datos estáticos de cada circuito
│   ├── data_collection.py     ← ETL FastF1 (histórico + snapshot post-quali)
│   ├── feature_engineering.py ← Pipeline de variables
│   └── predict_lambda.py      ← Handler AWS Lambda (solo XGBoost, sin TabNet)
│
├── models/                    ← Generado automáticamente al entrenar
│   ├── xgboost_f1_winner.pkl
│   ├── tabnet_model.zip
│   ├── label_encoders.pkl
│   └── scaler.pkl
│
└── data/                      ← Generado automáticamente al descargar datos
    ├── raw/                   ← race_results_raw.csv
    └── processed/             ← feature_importance.csv, historical_model_performance.csv
```

---

## Quick start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Configure AWS locally

```bash
aws configure
# Enter: Access Key ID, Secret Access Key, region (e.g. eu-west-1), output format (json)
```

### 3 · Set your S3 bucket

Edit `config.py` or set the environment variable:

```bash
export S3_BUCKET=your-bucket-name
```

### 4 · Collect historical data & train the model

```bash
# Descarga 2024-2025 via FastF1, entrena XGBoost + TabNet, guarda en /models
python train.py

# También sube artefactos XGBoost + métricas a S3
python train.py --upload-s3
```

### 5 · Predict after qualifying (Saturday)

```bash
# Predice ganador de la Ronda 5, temporada 2026
python predict.py --round 5

# Sin subir a S3 (guarda CSV localmente)
python predict.py --round 5 --no-upload

# Descarga modelo de S3 primero (máquina sin modelo local)
python predict.py --round 5 --from-s3
```

### 6 · Record actual winner (Sunday evening)

```bash
python predict.py --round 5 --record-result
```

This fills the `actual_winner` column in S3 `history.csv` so accuracy metrics
are available in Looker Studio.

---

## Features used by the model

| Category | Feature | Description |
|---|---|---|
| **Qualifying** | `grid_position` | Starting grid slot (1–22) |
| **Qualifying** | `quali_gap_to_pole_s` | Time gap to pole in seconds |
| **Clean-air pace** | `fp2_long_run_pace_gap_s` | Gap to fastest FP2 median lap (race-pace proxy on slick compounds) |
| **Clean-air pace** | `fp3_gap_to_best_s` | Gap to fastest FP3 lap |
| **Weather** | `rain_probability` | Fraction of qualifying session with active rainfall (0–1) |
| **Weather** | `is_wet_qualifying` | Binary: 1 if qualifying was wet |
| **Weather** | `track_temp_c` | Mean track temperature during qualifying (°C) |
| **Weather** | `humidity_pct` | Mean relative humidity (%) |
| **Weather** | `wind_speed_ms` | Mean wind speed (m/s) |
| **Driver form** | `driver_avg_finish_l3` | Rolling average finish position – last 3 races |
| **Driver form** | `driver_win_rate_l5` | Win rate over last 5 races |
| **Driver form** | `driver_wet_win_rate` | Win rate specifically in wet-condition races (last 10) |
| **Circuit history** | `driver_best_finish_circuit` | Best historical finish at this circuit |
| **Circuit history** | `driver_avg_finish_circuit` | Average historical finish at this circuit |
| **Circuit (static)** | `overtake_difficulty` | 1 (Monza) – 5 (Monaco): how hard it is to overtake |
| **Circuit (static)** | `drs_zones` | Number of DRS activation zones |
| **Circuit (static)** | `avg_safety_car_prob` | Historical SC/VSC probability at this circuit |
| **Championship** | `driver_champ_pos` / `driver_champ_points` | Driver standings before race |
| **Championship** | `constructor_champ_pos` / `constructor_champ_points` | Constructor standings before race |
| **Context** | `race_number` | Round number in the season |
| **Encoding** | `circuit_encoded` / `constructor_encoded` | Label-encoded identifiers |

---

## AWS infrastructure

### S3 bucket layout

```
s3://your-bucket/
├── predictions/
│   └── history.csv          ← appended after every race weekend
├── models/
│   ├── xgboost_f1_winner.pkl
│   └── label_encoders.pkl
└── athena-results/          ← Athena query output location
```

### Athena table

Create the table once (after your first upload):

```python
from src.aws_utils import create_athena_table_if_not_exists
create_athena_table_if_not_exists()
```

Then point **Looker Studio** at the Athena data source using the
`f1_predictions.prediction_history` table.

### IAM permissions needed

| Resource | Permissions |
|---|---|
| S3 (bucket) | `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` |
| Athena | `athena:StartQueryExecution`, `athena:GetQueryExecution`, `athena:GetQueryResults` |
| Glue (Athena catalogue) | `glue:GetTable`, `glue:CreateTable` |

---

## Model details

- **Algorithm**: XGBoost binary classifier (`scale_pos_weight=21` to handle 1 winner / 22 drivers, 11 teams in 2026)
- **Validation**: Time-series leave-one-season-out cross-validation (no data leakage)
- **Key metrics tracked**: ROC-AUC, Average Precision, Log Loss
- **Output**: `win_probability` (0–1) per driver → ranked as `predicted_rank`

---

## Credentials — security notes

- **Local**: credentials live in `~/.aws/credentials` — never hard-code keys in `.py` files.
- **Lambda / EC2**: attach an IAM Role with the permissions above — no keys needed.
- **`.gitignore`**: make sure `data/`, `models/`, and `.fastf1_cache/` are excluded.
