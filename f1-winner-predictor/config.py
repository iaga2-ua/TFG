"""
config.py
=========
Configuración central del predictor F1 2026.
"""

import os
from pathlib import Path

# ─── PATHS ───────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent

# En Lambda /var/task es de sólo lectura; usamos /tmp para datos y caché
_IS_LAMBDA     = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
_WRITABLE_ROOT = Path("/tmp") if _IS_LAMBDA else PROJECT_ROOT

DATA_DIR       = _WRITABLE_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = _WRITABLE_ROOT / "models"
CACHE_DIR      = _WRITABLE_ROOT / ".fastf1_cache"

for _dir in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── SEASONS ─────────────────────────────────────────────────────────────────
TRAIN_SEASONS   = [2023, 2024, 2025, 2026]   # 2026: solo carreras ya disputadas
CURRENT_SEASON  = 2026

# ─── FASTF1 ──────────────────────────────────────────────────────────────────
FASTF1_CACHE    = str(CACHE_DIR)

# ─── FEATURE COLUMNS (22 Drivers Compatible) ─────────────────────────────────
FEATURE_COLS = [
    "grid_position", "quali_gap_to_pole_s",
    "fp2_long_run_pace_gap_s", "fp3_gap_to_best_s",
    # Sprint Race result (disponible antes de la clasificación en sprint weekends).
    # 0 = GP sin sprint (ninguna carrera sprint se disputó).
    # 1-8 = posición real en el Sprint Race → proxy de ritmo en carrera real.
    "sprint_race_position",
    "rain_probability", "is_wet_qualifying",
    "track_temp_c", "humidity_pct", "wind_speed_ms",
    "driver_champ_pos", "driver_champ_points",
    "constructor_champ_pos", "constructor_champ_points",
    "driver_avg_finish_l3", "driver_win_rate_l5", "driver_wet_win_rate",
    "driver_best_finish_circuit", "driver_avg_finish_circuit",
    # Forma en la temporada actual (evita sesgo por dominancia histórica de un piloto)
    "driver_current_season_win_rate", "driver_races_since_last_win",
    # Circuit metadata (FastF1 dinámico)
    "track_length_km", "corner_count",
    # Circuit metadata (tabla estática)
    "overtake_difficulty", "drs_zones", "avg_safety_car_prob",
    "race_number", "circuit_encoded", "constructor_encoded"
]

TARGET_COL = "is_winner" 

# ─── OPTUNA ─────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS = 30   # Número de trials por defecto para la búsqueda de hiperparámetros

# ─── MODELOS ─────────────────────────────────────────────────────────────────
# XGBoost (Cloud)
MODEL_FILE     = MODELS_DIR / "xgboost_f1_winner.pkl"
ENCODER_FILE   = MODELS_DIR / "label_encoders.pkl"

# TabNet (Local)
TABNET_MODEL_PATH        = MODELS_DIR / "tabnet_model"
SCALER_FILE              = MODELS_DIR / "scaler.pkl"
TABNET_TEMPERATURE_FILE  = MODELS_DIR / "tabnet_temperature.pkl"  # Temperature Scaling (T >= 1)
XGB_TEMPERATURE_FILE     = MODELS_DIR / "xgb_temperature.pkl"     # Temperature Scaling XGBoost

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    # scale_pos_weight=5: corrección parcial del desbalance 1:21.
    # El valor completo (21) sobreconcentra las probabilidades en el piloto
    # más dominante (e.g. ANT con 100% win-rate en 2026). Con 5 el modelo
    # sigue favoreciendo al ganador esperado pero deja spread real en P2-P5.
    "scale_pos_weight": 5,
    # base_score=1/22: prior natural (1 ganador entre 22 pilotos).
    # El default (0.5) es un prior absurdamente alto que infla las predicciones
    # de pilotos con pocos datos (pilotos nuevos como ANT con solo 2 carreras).
    "base_score": 0.045,
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
S3_HISTORY_KEY     = "predictions/history.csv"       # historial unificado (input de Athena)
S3_MODEL_KEY       = "models/xgboost_f1_winner.pkl"  # artefacto XGBoost para Lambda
S3_ENCODER_KEY     = "models/label_encoders.pkl"     # encoders compartidos con Lambda
S3_XGB_TEMPERATURE_KEY = "models/xgb_temperature.pkl"  # Temperature Scaling XGBoost
S3_IMPORTANCE_KEY    = "metrics/feature_importance.csv"
S3_PERFORMANCE_KEY   = "metrics/historical_performance.csv"
S3_MODEL_ACCURACY_KEY = "metrics/model_accuracy/model_accuracy.csv"  # métricas Brier+MAE → Athena
S3_RACE_RESULTS_KEY  = "data/race_results_raw.csv"    # resultados históricos de carrera (features for Lambda)

# ─── ATHENA (motor SQL sobre S3 → Looker Studio) ──────────────────────────────────
ATHENA_DATABASE          = os.environ.get("ATHENA_DATABASE",          "f1_predictions")
ATHENA_TABLE             = os.environ.get("ATHENA_TABLE",             "race_predictions")
ATHENA_TABLE_IMPORTANCE  = os.environ.get("ATHENA_TABLE_IMPORTANCE",  "feature_importance")
ATHENA_TABLE_ACCURACY    = os.environ.get("ATHENA_TABLE_ACCURACY",    "model_accuracy")
ATHENA_OUTPUT_LOC        = f"s3://{S3_BUCKET}/athena-results/"