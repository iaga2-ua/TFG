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
    download_model_artefacts,
    read_history_csv,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cache warm-start
_XGB_MODEL = None
_ENCODERS  = None

def _load_models():
    """Descarga y carga modelos usando /tmp como puente."""
    global _XGB_MODEL, _ENCODERS
    if _XGB_MODEL is not None:
        return _XGB_MODEL, _ENCODERS

    tmp = Path("/tmp")
    model_tmp = tmp / "xgboost_f1_winner.pkl"
    encoder_tmp = tmp / "label_encoders.pkl"

    # Redirección temporal de rutas de config para usar download_model_artefacts
    import config as cfg
    old_m, old_e = cfg.MODEL_FILE, cfg.ENCODER_FILE
    cfg.MODEL_FILE, cfg.ENCODER_FILE = model_tmp, encoder_tmp
    
    try:
        download_model_artefacts()
        with open(model_tmp, "rb") as f: _XGB_MODEL = pickle.load(f)
        with open(encoder_tmp, "rb") as f: _ENCODERS = pickle.load(f)
        logger.info("✅ Modelos cargados exitosamente en Lambda.")
    finally:
        # Restaurar rutas originales
        cfg.MODEL_FILE, cfg.ENCODER_FILE = old_m, old_e

    return _XGB_MODEL, _ENCODERS

def predict(year: int, round_num: int, upload: bool = True) -> dict:
    logger.info("Iniciando prediccion XGBoost: %d Ronda %d", year, round_num)

    df_live = fetch_qualifying_snapshot(year, round_num)
    if df_live.empty:
        raise ValueError(f"No se encontraron datos de clasificacion para {year} R{round_num}")

    xgb_model, encoders = _load_models()

    try:
        history_df = read_history_csv()
    except Exception as exc:
        logger.warning("Sin historial en S3, usando DF vacio: %s", exc)
        history_df = pd.DataFrame()

    X     = apply_features(df_live, encoders, history_df=history_df)
    probs = xgb_model.predict_proba(X)[:, 1]

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