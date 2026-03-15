"""
predict_lambda.py -- Handler de AWS Lambda
==========================================
Flujo (nube, independiente de local):
  1. EventBridge lo activa el sabado despues de la clasificacion.
  2. Descarga XGBoost de S3 (subido por train.py).
  3. Obtiene datos de clasificacion via FastF1.
  4. Predice el ganador (piloto con mayor probabilidad).
  5. Guarda una fila en S3: predictions/history.csv
     (predicted_winner_xgb + win_prob_xgboost).
"""

import io
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CURRENT_SEASON, MODEL_FILE, ENCODER_FILE
from src.data_collection import fetch_qualifying_snapshot
from src.feature_engineering import apply_features
from src.aws_utils import (
    append_to_history_csv,
    download_race_results,
    sync_to_sheets,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cache warm-start
_XGB_MODEL    = None
_ENCODERS     = None
_RACE_RESULTS = None  # race_results_raw.csv descargado de S3

def _load_models():
    """Descarga modelos desde S3 a /tmp y los carga."""
    global _XGB_MODEL, _ENCODERS
    if _XGB_MODEL is not None:
        return _XGB_MODEL, _ENCODERS

    import boto3
    from config import S3_BUCKET, S3_MODEL_KEY, S3_ENCODER_KEY, AWS_REGION

    s3 = boto3.client("s3", region_name=AWS_REGION)
    model_tmp   = Path("/tmp/xgboost_f1_winner.pkl")
    encoder_tmp = Path("/tmp/label_encoders.pkl")

    logger.info("Descargando modelos desde s3://%s ...", S3_BUCKET)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY,   str(model_tmp))
    s3.download_file(S3_BUCKET, S3_ENCODER_KEY, str(encoder_tmp))

    with open(model_tmp,   "rb") as f: _XGB_MODEL = pickle.load(f)
    with open(encoder_tmp, "rb") as f: _ENCODERS  = pickle.load(f)
    logger.info("✅ Modelos cargados exitosamente en Lambda.")
    return _XGB_MODEL, _ENCODERS

def _load_race_results() -> pd.DataFrame:
    """Descarga race_results_raw.csv de S3 (warm-cached por invocación)."""
    global _RACE_RESULTS
    if _RACE_RESULTS is not None:
        return _RACE_RESULTS
    logger.info("Descargando resultados históricos de carrera desde S3...")
    _RACE_RESULTS = download_race_results()
    logger.info("  -> %d filas cargadas.", len(_RACE_RESULTS))
    return _RACE_RESULTS

def predict(year: int, round_num: int, upload: bool = True) -> dict:
    logger.info("Iniciando prediccion XGBoost: %d Ronda %d", year, round_num)

    df_live = fetch_qualifying_snapshot(year, round_num)
    if df_live.empty:
        raise ValueError(f"No se encontraron datos de clasificacion para {year} R{round_num}")

    xgb_model, encoders = _load_models()

    # Resultados históricos de carrera: base para calcular standings, rolling stats, etc.
    race_results_df = _load_race_results()

    X     = apply_features(df_live, encoders, history_df=race_results_df, year=year, round_num=round_num)
    probs = xgb_model.predict_proba(X)[:, 1]

    # Normalizar entre los N pilotos para que las probabilidades sumen 1
    # y sean interpretables como cuota relativa de victoria.
    # XGBoost predice cada piloto de forma independiente (clasificacion binaria),
    # por lo que los valores brutos no suman 1 y pueden ser engañosamente altos.
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs = probs / probs_sum

    best_idx    = int(np.argmax(probs))
    winner_abbr = df_live.iloc[best_idx]["driver_abbr"]
    winner_prob = float(probs[best_idx])
    event_name  = df_live.iloc[0]["event_name"]
    circuit     = df_live.iloc[0]["circuit"]

    result_row = {
        "year":                 year,
        "round":                round_num,
        "event_name":           event_name,
        "circuit":              circuit,
        "predicted_winner_xgb": winner_abbr,
        "win_prob_xgboost":     winner_prob,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if upload:
        append_to_history_csv(result_row)
        logger.info("Resultado XGBoost guardado en S3: %s (%.1f%%)",
                    winner_abbr, winner_prob * 100)
        sync_to_sheets()

    return {
        "statusCode": 200,
        "body": {
            "year":             year,
            "round":            round_num,
            "event":            event_name,
            "predicted_winner": winner_abbr,
            "win_probability":  f"{winner_prob:.1%}",
        },
    }

def lambda_handler(event, context):
    """Handler oficial para AWS Lambda."""
    try:
        if "body" in event and isinstance(event["body"], str):
            payload = json.loads(event["body"])
        else:
            payload = event

        year      = int(payload.get("year", CURRENT_SEASON))
        round_num = int(payload.get("round"))
        upload    = payload.get("upload", True)

        return predict(year, round_num, upload)

    except Exception as exc:
        logger.error("Error critico en Lambda: %s", exc)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(exc)}),
        }