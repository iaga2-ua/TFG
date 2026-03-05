"""
config.py
=========
Configuración central del predictor F1 2026.
"""

import os
from pathlib import Path

# ─── PATHS ───────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
CACHE_DIR      = PROJECT_ROOT / ".fastf1_cache"

for _dir in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── SEASONS ─────────────────────────────────────────────────────────────────
# Recomendación: Incluir 2023 si tienes los datos para dar más volumen a TabNet
TRAIN_SEASONS   = [2023, 2024, 2025] 
CURRENT_SEASON  = 2026

# ─── FASTF1 ──────────────────────────────────────────────────────────────────
FASTF1_CACHE    = str(CACHE_DIR)

# ─── FEATURE COLUMNS (22 Drivers Compatible) ─────────────────────────────────
FEATURE_COLS = [
    "grid_position", "quali_gap_to_pole_s",
    "fp2_long_run_pace_gap_s", "fp3_gap_to_best_s",
    "rain_probability", "is_wet_qualifying",
    "track_temp_c", "humidity_pct", "wind_speed_ms",
    "driver_champ_pos", "driver_champ_points",
    "constructor_champ_pos", "constructor_champ_points",
    "driver_avg_finish_l3", "driver_win_rate_l5", "driver_wet_win_rate",
    "driver_best_finish_circuit", "driver_avg_finish_circuit",
    # Circuit metadata (FastF1 dinámico)
    "track_length_km", "corner_count",
    # Circuit metadata (tabla estática)
    "overtake_difficulty", "drs_zones", "avg_safety_car_prob",
    "race_number", "circuit_encoded", "constructor_encoded"
]

TARGET_COL = "is_winner" 

# ─── MODELOS ─────────────────────────────────────────────────────────────────
# XGBoost (Cloud)
MODEL_FILE     = MODELS_DIR / "xgboost_f1_winner.pkl"
ENCODER_FILE   = MODELS_DIR / "label_encoders.pkl"

# TabNet (Local)
TABNET_MODEL_PATH = MODELS_DIR / "tabnet_model"
SCALER_FILE       = MODELS_DIR / "scaler.pkl"

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    # Ajuste para 22 pilotos: 1 ganador vs 21 perdedores
    "scale_pos_weight": 21, 
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

# ─── AWS & LOOKER STUDIO ─────────────────────────────────────────────────────
AWS_REGION     = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
AWS_PROFILE    = os.environ.get("AWS_PROFILE", "default")
S3_BUCKET      = os.environ.get("S3_BUCKET", "f1-winner-predictor-2026")

# Nombre de la Lambda que ejecuta XGBoost (inferencia en la nube)
LAMBDA_FUNCTION_NAME = os.environ.get("LAMBDA_FUNCTION_NAME", "f1-winner-predictor")

# ─── S3 KEYS ────────────────────────────────────────────────────────────────────
S3_HISTORY_KEY     = "predictions/history.csv"     # historial unificado (input de Athena)
S3_MODEL_KEY       = "models/xgboost_f1_winner.pkl" # artefacto XGBoost para Lambda
S3_ENCODER_KEY     = "models/label_encoders.pkl"    # encoders compartidos con Lambda
S3_IMPORTANCE_KEY  = "metrics/feature_importance.csv"
S3_PERFORMANCE_KEY = "metrics/historical_performance.csv"

# ─── ATHENA (motor SQL sobre S3 → Looker Studio) ──────────────────────────────────
ATHENA_DATABASE          = os.environ.get("ATHENA_DATABASE",          "f1_predictions")
ATHENA_TABLE             = os.environ.get("ATHENA_TABLE",             "race_predictions")
ATHENA_TABLE_IMPORTANCE  = os.environ.get("ATHENA_TABLE_IMPORTANCE",  "feature_importance")
ATHENA_OUTPUT_LOC        = f"s3://{S3_BUCKET}/athena-results/"