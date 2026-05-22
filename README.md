<div align="center">

# 🏎️ F1 Winner Predictor 2026

**Pre-race Formula 1 winner prediction powered by XGBoost and TabNet, deployed on AWS Lambda with live Looker Studio dashboards.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-F97316?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![AWS Lambda](https://img.shields.io/badge/AWS-Lambda-FF9900?logo=awslambda&logoColor=white)](https://aws.amazon.com/lambda/)
[![Docker](https://img.shields.io/badge/Docker-containerised-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Looker Studio](https://img.shields.io/badge/Looker_Studio-live_dashboard-4285F4?logo=googlelookerstudio&logoColor=white)](https://datastudio.google.com/reporting/e898a7b0-6663-46d0-853b-8c2083fab69c)

[**📊 Live Dashboard**](https://datastudio.google.com/reporting/e898a7b0-6663-46d0-853b-8c2083fab69c)

</div>

---

## 📈 Results at a glance

| Model | Overall accuracy | 2026 live test (R01–R04) | Deployment |
|:---|:---:|:---:|:---|
| **XGBoost** | **95.9%** (71 / 74 races) | **4 / 4 — 100%** | AWS Lambda |
| TabNet | 70.3% (52 / 74 races) | 1 / 4 — 25% | Local / Docker |

> Predictions are based solely on Saturday qualifying data. Models are trained on 2023–2025 and evaluated on 2026 races without retraining.

---

## 🏗️ Architecture

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
    (Looker Studio       (available for
      active source)     ad-hoc SQL queries)
```

---

## 📁 Project structure

```text
f1-winner-predictor/
├── config.py                  # Central configuration (paths, features, parameters)
├── train.py                   # XGBoost + TabNet training with Optuna
├── predict.py                 # Local TabNet inference + result logging
├── predict_proba_all.py       # Full win-probability table (XGBoost + TabNet, no S3 writes)
├── requirements.txt           # Local/Docker environment dependencies
├── requirements-lambda.txt    # Lambda container dependencies
├── Dockerfile                 # Image for local training and prediction
├── Dockerfile.lambda          # Image for AWS Lambda (ECR)
├── Dockerfile.layer           # Lambda dependency layer
├── src/
│   ├── data_collection.py     # Downloads FastF1 data (quali, race, weather)
│   ├── feature_engineering.py # Builds features for XGBoost and TabNet
│   ├── aws_utils.py           # S3 and Google Sheets sync
│   ├── predict_lambda.py      # AWS Lambda handler (XGBoost inference)
│   └── circuit_metadata.py    # Static circuit metadata table
└── models/                    # Local artefacts (gitignored, managed via S3)
    ├── xgboost_f1_winner.pkl
    ├── label_encoders.pkl
    ├── tabnet_model.zip
    └── scaler.pkl
```

---

## 🔢 Features (29 variables)

| Category | Feature | Description |
|:---|:---|:---|
| **Qualifying** | `grid_position` | Grid position (Saturday, before penalties) |
| **Qualifying** | `quali_gap_to_pole_s` | Gap to pole in seconds — `min(Q1, Q2, Q3)` across all sessions |
| **Practice** | `fp2_long_run_pace_gap_s` | Gap to the best long-run pace in FP2 |
| **Practice** | `fp3_gap_to_best_s` | Gap to the fastest time in FP3 |
| **Sprint** | `sprint_race_position` | Sprint Race position (0 = standard weekend) |
| **Weather** | `rain_probability` | Fraction of qualifying session with rain (0–1) |
| **Weather** | `is_wet_qualifying` | Binary: wet qualifying session |
| **Weather** | `track_temp_c`, `humidity_pct`, `wind_speed_ms` | Conditions during qualifying |
| **Championship** | `driver_champ_pos`, `driver_champ_points` | Driver standings before the race |
| **Championship** | `constructor_champ_pos`, `constructor_champ_points` | Constructor standings |
| **Driver history** | `driver_avg_finish_l3` | Average finishing position over the last 3 races |
| **Driver history** | `driver_win_rate_l5` | Win rate over the last 5 races |
| **Driver history** | `driver_wet_win_rate` | Win rate in wet races (last 10) |
| **Driver history** | `driver_current_season_win_rate` | Win rate in the current season |
| **Driver history** | `driver_races_since_last_win` | Races since last win, capped at 50 |
| **Circuit history** | `driver_best_finish_circuit`, `driver_avg_finish_circuit` | Best and average result at this circuit over the last 3 years |
| **Circuit (dynamic)** | `track_length_km`, `corner_count` | Track length and corner count via FastF1 |
| **Circuit (static)** | `overtake_difficulty`, `drs_zones`, `avg_safety_car_prob` | Overtaking difficulty, DRS zones and historical safety car probability |
| **Context** | `race_number` | Round number within the season |
| **Encoded** | `circuit_encoded`, `constructor_encoded` | Label-encoded circuit and constructor |

---

## 🤖 Models

### XGBoost — cloud deployment via AWS Lambda

- Binary classifier (`is_winner = 1` for the race winner)
- `scale_pos_weight = 5`: partial class-imbalance correction (full value of 21 over-concentrates probabilities)
- `base_score = 0.045` (= 1/22): natural win prior to avoid inflated estimates for sparse drivers
- Hyperparameters tuned with **Optuna** (ROC-AUC, 5-fold stratified CV)
- Trained with **recency weights** `{2023:1, 2024:1, 2025:2, 2026:2}` via `sample_weight`
- Artefacts in S3: `xgboost_f1_winner.pkl`, `label_encoders.pkl`, `xgb_temperature.pkl`

### TabNet — local inference

- Tabular neural network with sequential attention mechanism (`pytorch-tabnet`)
- Pre-standardised with `StandardScaler`
- Hyperparameters tuned with **Optuna** (ROC-AUC, 20% validation split)
- Trained with **recency oversampling** `{2023:1, 2024:1, 2025:2, 2026:2}` via row duplication
- Artefacts: `tabnet_model.zip`, `scaler.pkl`, `tabnet_temperature.pkl`

> **Why two models?** TabNet requires PyTorch (~1 GB of dependencies), which is incompatible with AWS Lambda's size constraints. XGBoost is lightweight (~50 MB) and deploys without issues. Running both models enables a direct architectural comparison.

### Probability calibration — Temperature Scaling

Both models are calibrated post-hoc with **Temperature Scaling** ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)).

**Why calibration is needed:**
1. Without it, the pole-sitter can receive ~90–99% win probability, far above the true rate (~40–45%)
2. XGBoost's 22 independent binary classifiers do not naturally sum to 100%

**How it works:** each model is trained on 80% of the data. On the remaining 20%, NLL is minimised with L-BFGS to find the optimal temperature $T$:

$$p_{\text{cal}} = \frac{1}{1 + e^{-z/T}}, \qquad p_{\text{norm}} = \frac{p_{\text{cal}}}{\displaystyle\sum_i p_{\text{cal},i}}$$

where $z = \log\frac{p}{1-p}$ are the logits extracted from the raw model output. $T > 1$ flattens probabilities and reduces overconfidence.

---

## ☁️ AWS Infrastructure

| Resource | Details |
|:---|:---|
| **Region** | `eu-west-1` |
| **S3 bucket** | `f1-winner-predictor-2026` |
| **Lambda function** | `f1-winner-predictor` (ECR image) |
| **ECR** | `606756239522.dkr.ecr.eu-west-1.amazonaws.com/f1-winner-predictor:latest` |
| **Athena** | DB: `f1_predictions` / Tables: `race_predictions`, `feature_importance` |

### S3 layout

```text
f1-winner-predictor-2026/
├── predictions/
│   └── history.csv                   # One row per race (XGBoost + TabNet merged)
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

## 📊 Google Sheets and Looker Studio

Predictions and metrics sync automatically to Google Sheets (linked as the Looker Studio data source):

| Tab | Content | Updated |
|:---|:---|:---|
| **Sheet1** | Full prediction history with 2026 results | Saturday (predict) and Monday (log result) |
| **feature_importance** | Per-feature importance for both models | On retrain with `--upload-s3` |
| **model_accuracy** | Cumulative accuracy, Brier Score and positional MAE | Monday (after each race) |

**[View live dashboard →](https://datastudio.google.com/reporting/e898a7b0-6663-46d0-853b-8c2083fab69c)**

---

## 📐 Evaluation metrics

Three complementary metrics are computed cumulatively after every race:

### Accuracy
$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{races completed}}$$

### Brier Score (lower is better, 0 = perfect)
$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$$

$p_i$ is the probability assigned to the predicted driver and $y_i \in \{0,1\}$ indicates whether they actually won. Penalises overconfident wrong predictions much more heavily than cautious ones.

### Positional MAE (lower is better, 0 = winner predicted)
$$\text{MAE}_{\text{pos}} = \frac{1}{N}\sum_{i=1}^{N}|f_i - 1|$$

$f_i$ is the finishing position of the driver the model predicted as winner. Zero if the model was right, four if the pick finished P5.

---

## 🚀 Workflow

### Friday (optional) — retrain models

```powershell
# Full retrain with Optuna tuning and S3 upload
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data --optimize

# Quick retrain without hyperparameter search
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data

# Single model
docker compose run --rm trainer python train.py --model xgboost --upload-s3
docker compose run --rm trainer python train.py --model tabnet
```

### Saturday — predict after qualifying

```powershell
# XGBoost via Lambda
aws lambda invoke --function-name f1-winner-predictor `
  --payload '{"year":2026,"round":5}' response.json --region eu-west-1

# TabNet via Docker
docker compose run --rm trainer python predict.py --round 5 --year 2026
```

### Saturday — full probability table (no S3 writes)

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

### Monday — log the actual result

```powershell
docker compose run --rm trainer python predict.py --round 5 --year 2026 --record-result
```

This downloads the race result from FastF1, writes `actual_winner` and accuracy columns to S3 and updates all three Google Sheets tabs.

---

## ⚙️ Deployment script (`deploy.ps1`)

| Mode | What it does |
|:---|:---|
| `models` | Retrains XGBoost + TabNet and uploads artefacts to S3 |
| `lambda` | Rebuilds Docker image, pushes to ECR and updates Lambda |
| `predict` | Invokes Lambda + runs local TabNet and saves probability snapshot |
| `all` | Runs `models`, `lambda` and `predict` in sequence |

```powershell
.\deploy.ps1 -Mode lambda                    # Redeploy Lambda only
.\deploy.ps1 -Mode models                    # Retrain and upload to S3
.\deploy.ps1 -Mode all -Refresh              # Full pipeline with fresh data
.\deploy.ps1 -Mode all -Refresh -Optimize    # Full pipeline with Optuna tuning
```

---

## 🏁 Sprint weekends

Sprint weekends have a different session schedule, which requires two fixes:

| Issue | Cause | Fix |
|:---|:---|:---|
| Missing FP2/FP3 | Sessions do not exist on sprint weekends | FP1 pace used as substitute for both columns |
| Understated standings | Sprint Race points omitted from `race_results_raw.csv` | Sprint Race (`"S"` session) points added to main-race points on ingestion |

| Feature | Without fix | With fix |
|:---|:---:|:---:|
| `fp2_long_run_pace_gap_s` | `0` for all drivers | Real gap from FP1 |
| `fp3_gap_to_best_s` | `0` for all drivers | Real gap from FP1 |
| `driver_champ_points` | Understated | Exact (main race + sprint) |
| `constructor_champ_points` | Understated | Exact (main race + sprint) |

---

## 🔧 Initial setup

### 1. Environment variables

Create `.env` at the project root:

```env
AWS_PROFILE=f1-developer
AWS_REGION=eu-west-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 2. Google Sheets credentials

Download the JSON for service account `f1-sheets-sync@f1-tfg.iam.gserviceaccount.com` and place it at:

```text
f1-winner-predictor/.google_credentials.json
```

> This file is in `.gitignore` and must never be committed.

### 3. Build the Docker image

```powershell
cd "TFG"
docker build -t f1-tabnet:latest .
```


