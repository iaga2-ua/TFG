# F1 Winner Predictor 2026

Predicts the winner of each Formula 1 Grand Prix based on Saturday qualifying data. Combines two machine learning models — **XGBoost** (deployed on AWS Lambda) and **TabNet** (local inference) — and synchronises predictions and metrics with Google Sheets for visualisation in Looker Studio.

---

## General architecture

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
     — active source)    ad-hoc SQL queries)
```

---

## Project structure

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
│   ├── __init__.py
│   ├── data_collection.py     # Downloads FastF1 data (quali, race, weather)
│   ├── feature_engineering.py # Builds features for XGBoost and TabNet
│   ├── aws_utils.py           # S3, Google Sheets sync (Athena configured for ad-hoc SQL)
│   ├── predict_lambda.py      # AWS Lambda handler (XGBoost inference)
│   └── circuit_metadata.py    # Static circuit metadata table
└── models/                    # Local artefacts (gitignored, managed via S3)
    ├── xgboost_f1_winner.pkl
    ├── label_encoders.pkl
    ├── tabnet_model.zip
    └── scaler.pkl
```

---

## Features used (28 variables)

| Category | Feature | Description |
| --- | --- | --- |
| **Qualifying** | `grid_position` | Grid position (Saturday, penalties not applied) |
| **Qualifying** | `quali_gap_to_pole_s` | Gap to pole in seconds. Computed as `min(Q1, Q2, Q3)` to capture each driver's actual fastest lap regardless of which session they were eliminated in |
| **Practice** | `fp2_long_run_pace_gap_s` | Gap to the best long-run pace in FP2 |
| **Practice** | `fp3_gap_to_best_s` | Gap to the fastest time in FP3 |
| **Sprint Race** | `sprint_race_position` | Sprint Race position (0 = non-sprint weekend). Proxy for real race pace, available before Saturday qualifying |
| **Weather** | `rain_probability` | Fraction of the session with rain (0–1) |
| **Weather** | `is_wet_qualifying` | Binary: wet qualifying session |
| **Weather** | `track_temp_c`, `humidity_pct`, `wind_speed_ms` | Conditions during qualifying |
| **Championship** | `driver_champ_pos`, `driver_champ_points` | Driver standings position and points before the race |
| **Championship** | `constructor_champ_pos`, `constructor_champ_points` | Constructor standings position and points |
| **Driver history** | `driver_avg_finish_l3` | Average finishing position over the last 3 races |
| **Driver history** | `driver_win_rate_l5` | Win rate over the last 5 races |
| **Driver history** | `driver_wet_win_rate` | Win rate in wet races (last 10) |
| **Driver history** | `driver_current_season_win_rate` | Win rate in the current season. Avoids bias from a driver's historical dominance in previous seasons |
| **Driver history** | `driver_races_since_last_win` | Races since last win (capped at 50). Penalises drivers who haven't won recently |
| **Circuit history** | `driver_best_finish_circuit`, `driver_avg_finish_circuit` | Best and average result at this circuit over the **last 3 years** (older history excluded to avoid obsolete wins distorting predictions) |
| **Circuit (FastF1)** | `track_length_km`, `corner_count` | Track length and corner count (dynamic) |
| **Circuit (static)** | `overtake_difficulty`, `drs_zones`, `avg_safety_car_prob` | Overtaking difficulty, DRS zones, historical safety car probability |
| **Context** | `race_number` | Round number within the season |
| **Encoded** | `circuit_encoded`, `constructor_encoded` | Label encoding of circuit and constructor |

---

## Models

### XGBoost (cloud — AWS Lambda)

- Binary classifier (`is_winner = 1` for the race winner)
- `scale_pos_weight = 5`: partial class-imbalance correction. The full value (21) over-concentrates probabilities on the favourite; 5 maintains a realistic spread across the top drivers.
- `base_score = 0.045` (= 1/22): natural win prior. XGBoost's default (0.5) artificially inflated estimates for drivers with sparse data.
- Hyperparameters optimised with **Optuna** (ROC-AUC, 5-fold stratified CV)
- Trained on 2023–2026 data with **recency weights** `{2023:1, 2024:1, 2025:2, 2026:2}` via `sample_weight`
- Artefacts in S3: `models/xgboost_f1_winner.pkl` + `models/label_encoders.pkl` + `models/xgb_temperature.pkl`

#### Probability calibration (Temperature Scaling)

XGBoost predicts each driver as an independent binary classifier. This causes two problems:

1. **Extreme confidence**: without calibration, the pole-sitter can receive ~90–99%, well above the true win rate from pole (~40–45%).
2. **Probabilities do not sum to 1**: the 22 independent classifiers' outputs need not sum to 100%.

**Temperature Scaling** (`train.py → _fit_temperature`): XGBoost is trained on **80%** of the data. The remaining **20%** is used to minimise NLL and find the optimal $T$, saved to `models/xgb_temperature.pkl`.

At inference time (`predict_lambda.py`, `predict.py`, `predict_proba_all.py`), logits are extracted ($z = \log\frac{p}{1-p}$), divided by $T$, passed through sigmoid, and finally normalised across the $N$ drivers:

$$p_{\text{cal}} = \frac{1}{1 + e^{-z/T}}, \quad p_{\text{norm}} = \frac{p_{\text{cal}}}{\sum_i p_{\text{cal},i}}$$

### TabNet (local)

- Tabular neural network with sequential attention mechanism (pytorch-tabnet)
- Same data and features as XGBoost
- Pre-standardised with `StandardScaler`
- Hyperparameters optimised with **Optuna** (ROC-AUC, 20% validation)
- Trained with **recency oversampling** `{2023:1, 2024:1, 2025:2, 2026:2}`: rows from recent seasons are duplicated in the training set. TabNet does not natively support `sample_weight`; row duplication is equivalent.
- Artefacts locally: `models/tabnet_model.zip` + `models/scaler.pkl` + `models/tabnet_temperature.pkl`

#### Probability calibration — TabNet

Neural networks tend to be **overconfident**: the final softmax produces probabilities close to 1 even under uncertainty. This is corrected with **Temperature Scaling**, the standard post-hoc calibration technique for neural networks ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)).

The idea is to divide the logits $z$ (pre-sigmoid) by a temperature $T \geq 1$ before computing the final probability:

$$p_{\text{cal}} = \sigma\!\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z/T}}$$

- $T = 1$: no change (original model)
- $T > 1$: flattens probabilities, reduces overconfidence
- $T < 1$: sharpens probabilities, increases confidence (uncommon in neural networks)

**How $T$ is learned** (`train.py → _fit_temperature`): TabNet is trained on **80%** of the data. On the remaining **20%** (calibration set), NLL is minimised with L-BFGS to find the optimal $T$, saved to `models/tabnet_temperature.pkl`.

The same mechanism is applied to **XGBoost** (`models/xgb_temperature.pkl`) with the same 80/20 split and the same `_fit_temperature` function.

**How it is applied**: at inference time, logits are extracted from raw probabilities ($z = \log\frac{p}{1-p}$), divided by $T$, and passed through sigmoid. If the temperature file is missing, $T = 1.0$ is used with a warning.

> **Rationale for the dual design**: TabNet requires PyTorch (~1 GB of dependencies), incompatible with AWS Lambda's practical size constraints. XGBoost is lightweight (~50 MB) and deploys without issues. Running both models allows direct comparison of their predictions.

---

## AWS Infrastructure

| Resource | Details |
| --- | --- |
| **Region** | `eu-west-1` |
| **S3 bucket** | `f1-winner-predictor-2026` |
| **Lambda** | `f1-winner-predictor` (ECR image) |
| **ECR** | `606756239522.dkr.ecr.eu-west-1.amazonaws.com/f1-winner-predictor:latest` |
| **Athena** | DB: `f1_predictions`, tables: `race_predictions`, `feature_importance` — configured for ad-hoc SQL queries; Looker Studio uses Google Sheets as the active source |

### S3 bucket structure

```text
f1-winner-predictor-2026/
├── predictions/
│   └── history.csv              # One row per race (XGB + TabNet merged)
├── models/
│   ├── xgboost_f1_winner.pkl
│   ├── label_encoders.pkl
│   ├── xgb_temperature.pkl      # Temperature Scaling for XGBoost
│   └── tabnet_temperature.pkl   # Temperature Scaling for TabNet
├── metrics/
│   ├── feature_importance.csv   # Feature importance by model
│   └── historical_performance.csv  # Predictions vs results (training data)
└── data/
    └── race_results_raw.csv     # Historical training data
```

### XGB + TabNet merge in history.csv

Lambda writes `predicted_winner_xgb` and the local script writes `predicted_winner_tab`. The `append_to_history_csv()` function merges both writes into the **same row** (indexed by `year` + `round`) without overwriting columns already written by the other model.

---

## Google Sheets + Looker Studio

Predictions and metrics are automatically synchronised with Google Sheets (ID: `1Jw7wo3bqC2IS9MmfSJe6T2waQyp7LTCMv4al7gwhtPI`):

| Tab | Content | When updated |
| --- | --- | --- |
| **Sheet1** | `history.csv` — 2026 predictions per race | When predicting (Saturday) and when logging the result (Monday) |
| **feature_importance** | Importance of each feature in XGBoost and TabNet | When retraining with `--upload-s3` |
| **model_accuracy** | Cumulative accuracy, Brier Score and positional MAE race by race | When logging the actual winner (Monday) |

### Looker Studio charts

| Chart | Source | Configuration |
| --- | --- | --- |
| **Predictions table** | Sheet1 | Dimensions: `event_name`, `predicted_winner_xgb`, `predicted_winner_tab`, `actual_winner`, `xgb_correct`, `tab_correct` / Metrics: `win_prob_xgboost`, `win_prob_tabnet` |
| **Feature importance bars** | `feature_importance` | Y dimension: `feature` / X metrics: `importance_xgboost` + `importance_tabnet` |
| **Accuracy line** | `model_accuracy` | X dimension: `race_label` / Y metrics: `xgb_accuracy_cumul` + `tab_accuracy_cumul` |
| **Brier Score line** | `model_accuracy` | X dimension: `race_label` / Y metrics: `xgb_brier_cumul` + `tab_brier_cumul` (↓ better, min 0) |
| **Positional MAE line** | `model_accuracy` | X dimension: `race_label` / Y metrics: `xgb_pos_mae_cumul` + `tab_pos_mae_cumul` (↓ better, min 0) |

---

## Evaluation metrics

Every time the actual winner is logged (`--record-result`), three cumulative metrics are computed and synchronised to compare XGBoost and TabNet:

### Cumulative accuracy

Fraction of races in which the model correctly predicted the winner:

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{races completed}}$$

Columns: `xgb_accuracy_cumul`, `tab_accuracy_cumul`.

### Cumulative Brier Score

Measures **probability calibration** — not just whether the model got it right, but how confident it was when it did or did not. Lower is better (0 = perfect, 1 = completely wrong with full confidence):

$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$$

Where $p_i$ is the probability assigned to the predicted driver and $y_i \in \{0, 1\}$ indicates whether they actually won.

**How it is computed** (`aws_utils.py → sync_model_accuracy_to_sheets`):

```python
# xgb_brier_i = (win_prob_xgboost - xgb_correct)²
df["xgb_brier"] = (df["win_prob_xgboost"] - df["xgb_correct"]) ** 2
df["xgb_brier_cumul"] = df["xgb_brier"].expanding().mean()
```

Columns: `xgb_brier`, `xgb_brier_cumul`, `tab_brier`, `tab_brier_cumul`.

### Cumulative positional MAE

Measures **how many positions the model was off** relative to P1. If the model predicted VER and VER finished P5, the error is 4. If the predicted driver won, the error is 0:

$$\text{MAE}_{\text{pos}} = \frac{1}{N}\sum_{i=1}^{N}|f_i - 1|$$

Where $f_i$ is the finishing position of the driver predicted as race winner.

**How it is obtained** (`predict.py → record_actual_result`): when logging the actual result, FastF1 downloads the race classification and looks up the finishing position of each model's predicted winner. That data is stored in `xgb_predicted_finish_pos` / `tab_predicted_finish_pos` inside `history.csv`.

```python
# Inside record_actual_result():
xgb_finish_pos = results[results["Abbreviation"] == xgb_pred]["Position"].values[0]
# Error_i = |final_pos - 1|  →  0 if won, 4 if P5, etc.
```

Columns: `xgb_predicted_finish_pos`, `xgb_pos_mae_cumul`, `tab_predicted_finish_pos`, `tab_pos_mae_cumul`.

---

## Initial setup (first run)

### 1. Environment variables

Create `.env` at the project root (`TFG/`):

```env
AWS_PROFILE=f1-developer
AWS_REGION=eu-west-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 2. Google Sheets credentials

Download the JSON for the service account `f1-sheets-sync@f1-tfg.iam.gserviceaccount.com` and place it at:

```text
f1-winner-predictor/.google_credentials.json
```

This file is in `.gitignore` and must never be committed to the repository.

### 3. Build the Docker image

```powershell
cd "TFG"
docker build -t f1-tabnet:latest .
```

---

## Weekly workflow

### Friday (optional) — Retrain models

Only needed when new races have been run or model improvements are desired.

```powershell
# Full training with hyperparameter optimisation and S3 upload:
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data --optimize

# Without optimisation (faster, uses default hyperparameters):
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data

# Single model:
docker compose run --rm trainer python train.py --model xgboost --upload-s3
docker compose run --rm trainer python train.py --model tabnet
```

On completion the following are automatically updated:

- XGBoost and encoders in S3 (available for Lambda)
- TabNet and scaler in `models/` locally
- `feature_importance` tab in Google Sheets

### After qualifying (Saturday) — Predict

**XGBoost via Lambda** (invoke manually after qualifying):

```powershell
aws lambda invoke --function-name f1-winner-predictor `
  --payload '{"year":2026,"round":2}' response.json --region eu-west-1
```

**TabNet via Docker** (manual, post-qualifying):

```powershell
docker compose run --rm trainer python predict.py --round 2 --year 2026
```

Both predictions are merged into a single row in S3 and synchronised with Google Sheets.

### After qualifying (Saturday) — Full probability table

To view the complete win-probability distribution for all drivers **without writing anything to S3**, use the `proba` service:

```powershell
docker compose run --rm proba --round 3
docker compose run --rm proba --round 3 --year 2026
```

The script loads local models (`models/`) and displays a table sorted by XGBoost probability. The table is automatically saved to `data/processed/proba_table_{year}_R{round:02d}.csv`:

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
  ...
[INFO] Table saved to: data/processed/proba_table_2026_R03.csv
```

> Each model's probabilities sum to 100% after normalising binary classifier outputs. TabNet spreads more probability across several drivers while XGBoost tends to concentrate it more on the favourite.

### After the race (Monday) — Log the actual result

```powershell
# FastF1 automatically downloads the winner from the API:
docker compose run --rm trainer python predict.py --round 2 --year 2026 --record-result
```

This:

1. Downloads the result from FastF1
2. Writes `actual_winner`, `xgb_correct`, `tab_correct`, `xgb_predicted_finish_pos` and `tab_predicted_finish_pos` to S3
3. Synchronises Sheet1 with the actual result
4. Updates the `model_accuracy` tab with cumulative accuracy, Brier Score and positional MAE
5. Saves `metrics/model_accuracy/model_accuracy.csv` to S3 for Athena

---

## Deployment (`deploy.ps1`)

`deploy.ps1` automates the full deployment pipeline. It accepts four modes:

| Mode | What it does |
| --- | --- |
| `models` | Retrains XGBoost + TabNet and uploads artefacts to S3 (Lambda untouched) |
| `lambda` | Rebuilds Docker image → pushes to ECR → updates Lambda function (no retraining) |
| `predict` | Invokes Lambda (XGBoost) + runs local TabNet, generates snapshot and probability CSV |
| `all` | `models` + `lambda` + `predict` in order |

```powershell
# Redeploy Lambda only (code changed, model unchanged):
.\deploy.ps1 -Mode lambda

# Retrain and upload models to S3 (no Lambda rebuild):
.\deploy.ps1 -Mode models

# Full pipeline with fresh data:
.\deploy.ps1 -Mode all -Refresh

# Full pipeline with hyperparameter optimisation:
.\deploy.ps1 -Mode all -Refresh -Optimize
```

> `.py` files are mounted as volumes during training, so local changes take effect immediately without rebuilding the trainer Docker image.

---

## Sprint weekends

On a sprint weekend the schedule differs from a standard race weekend:

| Session | Normal GP | Sprint GP |
| --- | --- | --- |
| Friday morning | FP1 | FP1 |
| Friday afternoon | FP2 | Sprint Qualifying (SQ) |
| Saturday morning | FP3 | Sprint Race (S) |
| Saturday afternoon | Qualifying (Q) | Qualifying (Q) |

This creates two structural problems for the model if not handled explicitly:

### 1. FP2/FP3 do not exist → fallback to FP1

`fetch_practice_pace` tries to load FP2 and FP3. On a sprint weekend neither session exists and FastF1 raises an exception. Without a fix, `fp2_long_run_pace_gap_s` and `fp3_gap_to_best_s` would be filled with `0` for **all** drivers after the `fillna(0)` in `apply_features`, erasing all practice-pace information and making the model unable to differentiate drivers by pace.

**Fix implemented** (`fetch_practice_pace`): when both sessions return empty, FP1 is loaded instead. The per-driver best-lap gaps from FP1 (relative to the fastest driver) are used as a substitute for both columns. It is a degraded but real proxy — FP1 contains actual pace data, unlike an artificial zero.

### 2. Sprint Race points not counted in the championship

`race_results_raw.csv` previously stored only main-race points. On a sprint weekend drivers earn additional points from the Sprint Race (up to 8 pts for P1 since 2023). If those points are omitted, `driver_champ_points` and `constructor_champ_points` are understated from that round onwards, contaminating standings for all subsequent races in the season.

**Fix implemented** (`fetch_season_results`): the `"S"` session (Sprint Race) is attempted for each round. If it exists, each driver's sprint points are added to their main-race points in the `points` column of `race_results_raw.csv`. Since `_championship_standings` and `_enrich_live_from_history` sum that column directly, standings are automatically correct without any changes to `feature_engineering.py`.

### Impact of the fix

| Feature | Without fix (sprint weekend) | With fix |
| --- | --- | --- |
| `fp2_long_run_pace_gap_s` | `0` for all drivers | Real gap from FP1 |
| `fp3_gap_to_best_s` | `0` for all drivers | Real gap from FP1 |
| `sprint_race_position` | `0` (no sprint) | Actual Sprint Race position |
| `driver_champ_points` | Understated | Exact (main race + sprint) |
| `constructor_champ_points` | Understated | Exact (main race + sprint) |

> **Note**: after any change to `data_collection.py`, regenerate `race_results_raw.csv` with `--refresh-data` so that historical data reflects the fix.

---

## Security

- AWS credentials go in `.env` (gitignored) or system environment variables
- Google credentials go in `.google_credentials.json` (gitignored)
- Model artefacts (`models/*.pkl`, `models/*.zip`) and data (`data/`) are in `.gitignore`
- The `package/` directory (Lambda build) is in `.gitignore`
- No credentials are hardcoded in source files
