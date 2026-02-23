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

# Aseguramos que el path incluya la raíz para importar config y src
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CURRENT_SEASON, MODEL_FILE, ENCODER_FILE
from src.data_collection import fetch_qualifying_snapshot
from src.feature_engineering import apply_features
from src.aws_utils import (
    append_to_history_csv,
    download_model_artefacts,
    read_history_csv,
)

# Configuración de Logging para AWS CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Caché en memoria (Warm Start)
_XGB_MODEL   = None
_ENCODERS    = None

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
    logger.info(f"🏎️ Iniciando predicción: {year} Ronda {round_num}")

    # 1. Obtener datos de la clasificación (Snapshot)
    df_live = fetch_qualifying_snapshot(year, round_num)
    if df_live.empty:
        raise ValueError(f"No se encontraron datos para {year} R{round_num}")

    # 2. Cargar cerebro (XGBoost)
    xgb_model, encoders = _load_models()

    # 3. Obtener historial para rolling features
    try:
        history_df = read_history_csv()
    except Exception as e:
        logger.warning(f"No se pudo leer el historial, se usará DF vacío: {e}")
        history_df = pd.DataFrame()

    # 4. Ingeniería de variables (Inferencia)
    X = apply_features(df_live, encoders, history_df=history_df)

    # 5. Predicción de probabilidades
    probs = xgb_model.predict_proba(X)[:, 1]

    # 6. Formatear resultados (Cumpliendo con las 22 plazas de 2026)
    results = df_live[[
        "year", "round", "event_name", "circuit",
        "driver_abbr", "team", "grid_position",
    ]].copy()
    
    results["win_prob_xgboost"] = probs.astype(float)
    results["win_prob_tabnet"] = 0.0  # Placeholder para el modelo local
    results["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()
    results["actual_winner"] = 0     # Se actualizará post-carrera
    
    results = results.sort_values("win_prob_xgboost", ascending=False).reset_index(drop=True)

    # 7. Persistencia
    if upload:
        append_to_history_csv(results)
        logger.info("☁️ Resultados sincronizados con el historial en S3.")

    # Top 3 para la respuesta rápida de la Lambda
    top3 = results.head(3).to_dict(orient="records")
    
    return {
        "statusCode": 200,
        "body": {
            "event": results.iloc[0]["event_name"],
            "winner_predicted": results.iloc[0]["driver_abbr"],
            "probability": f"{results.iloc[0]['win_prob_xgboost']:.2%}",
            "top3": top3
        }
    }

def lambda_handler(event, context):
    """Handler oficial para AWS Lambda."""
    try:
        # Manejar si el evento viene de API Gateway o invocación directa
        if "body" in event and isinstance(event["body"], str):
            payload = json.loads(event["body"])
        else:
            payload = event

        year = int(payload.get("year", CURRENT_SEASON))
        round_num = int(payload.get("round"))
        upload = payload.get("upload", True)

        response = predict(year, round_num, upload)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response)
        }

    except Exception as e:
        logger.error(f"❌ Error crítico: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }