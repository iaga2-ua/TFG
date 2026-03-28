"""
predict.py  (raiz del proyecto)
================================
Ejecucion local de inferencia post-clasificacion (Sabado).

Flujo completo del sistema:
  [Entrenamiento - local]
    python train.py --upload-s3        <- entrena XGBoost + TabNet con los mismos datos
                                          sube artefactos XGBoost a S3

  [Inferencia - nube]
    AWS Lambda (predict_lambda.py)     <- descarga modelo XGBoost de S3
                                          calcula probabilidades
                                          guarda resultado en S3 (history.csv)

  [Inferencia - local]  <-- ESTE SCRIPT
    python predict.py --round N        <- carga TabNet local
                                          calcula probabilidades TabNet
                                          sube resultado a S3 (solo win_prob_tabnet)

  [Visualizacion]
    S3 history.csv -> Athena -> Looker Studio

Uso:
    python predict.py --round 1                  # inferencia TabNet local
    python predict.py --round 5 --no-upload      # no sube resultados a S3
    python predict.py --round 5 --record-result  # registra ganador real (post-carrera)
"""

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CURRENT_SEASON, ENCODER_FILE, FASTF1_CACHE, MODEL_FILE, PROCESSED_DIR, RAW_DIR,
    TABNET_MODEL_PATH, SCALER_FILE, TABNET_TEMPERATURE_FILE, XGB_TEMPERATURE_FILE,
)
from src.data_collection import fetch_qualifying_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --- Carga de modelos --------------------------------------------------------

def load_tabnet():
    """Carga TabNet, Scaler, temperatura y los encoders de etiquetas."""
    tabnet_zip = TABNET_MODEL_PATH.with_suffix(".zip")
    if not (TABNET_MODEL_PATH.exists() or tabnet_zip.exists()) or not SCALER_FILE.exists():
        logger.warning("TabNet o Scaler no encontrados en /models. Ejecuta train.py primero.")
        return None, None, {}, 1.0
    from pytorch_tabnet.tab_model import TabNetClassifier
    tab = TabNetClassifier()
    tab.load_model(str(TABNET_MODEL_PATH) + ".zip")
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    encoders = {}
    if ENCODER_FILE.exists():
        with open(ENCODER_FILE, "rb") as f:
            encoders = pickle.load(f)
    else:
        logger.warning("ENCODER_FILE no encontrado: circuit_encoded/constructor_encoded seran -1.")
    temperature = 1.0
    if TABNET_TEMPERATURE_FILE.exists():
        with open(TABNET_TEMPERATURE_FILE, "rb") as f:
            temperature = pickle.load(f)
        logger.info("Temperature Scaling cargado: T=%.4f", temperature)
    else:
        logger.warning("tabnet_temperature.pkl no encontrado — usando T=1.0 (sin calibrar). "
                       "Reentrena con train.py para generarlo.")
    logger.info("TabNet + Scaler + encoders cargados.")
    return tab, scaler, encoders, temperature


# --- Historial para rolling features ----------------------------------------

def load_history() -> pd.DataFrame:
    """
    Carga el historial de carreras para rolling features (driver_avg_finish_l3, etc.).
    Siempre usa el CSV local: history.csv en S3 solo tiene una fila por carrera
    y no contiene resultados individuales por piloto.
    """
    local = RAW_DIR / "race_results_raw.csv"
    if local.exists():
        logger.info("Historial local cargado: %s", local)
        return pd.read_csv(local)
    logger.warning("race_results_raw.csv no encontrado: rolling features seran NaN.")
    return pd.DataFrame()


# --- Nucleo de prediccion local (TabNet) -------------------------------------

def predict_race(
    year: int,
    round_num: int,
    upload: bool = True,
) -> dict:
    """
    Inferencia local con TabNet.
    Calcula el ganador predicho y guarda una fila en S3 (solo win_prob_tabnet).
    Lambda escribe win_prob_xgboost de forma independiente.
    """
    from src.feature_engineering import apply_features

    logger.info("Iniciando prediccion local (TabNet): %d Round %d", year, round_num)

    df_live = fetch_qualifying_snapshot(year, round_num)

    tab_model, scaler, encoders, temperature = load_tabnet()
    if tab_model is None:
        raise RuntimeError(
            "TabNet no disponible. Ejecuta 'python train.py --model tabnet' primero."
        )

    history_df = load_history()
    X          = apply_features(df_live, encoders, history_df=history_df, year=year, round_num=round_num)
    X_scaled   = scaler.transform(X)

    # Obtener logits internos y aplicar Temperature Scaling antes del sigmoid
    # p_cal = sigmoid(logit / T),  T > 1 reduce la sobreconfianza
    raw_probs = tab_model.predict_proba(X_scaled)[:, 1]
    eps = 1e-7
    raw_probs = np.clip(raw_probs, eps, 1 - eps)
    logits    = np.log(raw_probs / (1 - raw_probs))
    cal_probs = 1.0 / (1.0 + np.exp(-logits / temperature))
    probs_tab = cal_probs

    # Normalizar entre los N pilotos del GP para que las probabilidades
    # sumen 1 y sean interpretables como cuota relativa de victoria.
    # TabNet hace clasificacion binaria independiente por piloto, por lo que
    # los valores brutos son bajos (no suman 1). El argmax es el mismo antes
    # y despues de normalizar, pero la cifra mostrada tiene sentido semantico.
    probs_sum = probs_tab.sum()
    if probs_sum > 0:
        probs_tab = probs_tab / probs_sum

    # --- Ganador predicho: piloto con mayor probabilidad ---
    best_idx    = int(np.argmax(probs_tab))
    winner_abbr = df_live.iloc[best_idx]["driver_abbr"]
    winner_prob = float(probs_tab[best_idx])
    event_name  = df_live.iloc[0]["event_name"]
    circuit     = df_live.iloc[0]["circuit"]

    result_row = {
        "year":                  year,
        "round":                 round_num,
        "event_name":            event_name,
        "circuit":               circuit,
        "predicted_winner_tab": winner_abbr,
        "win_prob_tabnet":      winner_prob,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _print_tabnet_winner(event_name, winner_abbr, winner_prob)

    # Guardar snapshot completo de probabilidades (todos los pilotos, ambos modelos)
    # para que predict_proba_all.py pueda reproducir exactamente la prediccion del sabado.
    _save_proba_snapshot(year, round_num, event_name, df_live, probs_tab, encoders, history_df)

    if upload:
        try:
            from src.aws_utils import append_to_history_csv, sync_to_sheets
            append_to_history_csv(result_row)
            logger.info("Ganador TabNet guardado en S3: %s (%.1f%%)",
                        winner_abbr, winner_prob * 100)
            sync_to_sheets()
        except Exception as exc:
            logger.error("Error al subir a S3: %s", exc)
            local_out = RAW_DIR / f"prediction_{year}_R{round_num:02d}.json"
            import json
            local_out.write_text(json.dumps(result_row, indent=2))
            logger.info("Guardado localmente: %s", local_out)

    return result_row


def _save_proba_snapshot(
    year: int,
    round_num: int,
    event_name: str,
    df_live: pd.DataFrame,
    tab_probs: np.ndarray,
    encoders: dict,
    history_df: pd.DataFrame,
) -> None:
    """
    Persiste el snapshot completo de probabilidades (XGBoost + TabNet) para todos
    los pilotos en data/processed/proba_snapshot_{year}_R{round_num:02d}.json.
    predict_proba_all.py lee este archivo para reproducir la prediccion del sabado.
    """
    import json
    from src.feature_engineering import apply_features

    drivers = df_live["driver_abbr"].tolist()

    # XGBoost: intentar obtener probabilidades invocando Lambda (modelo en S3)
    xgb_probs_dict: dict[str, float] = {}
    try:
        import boto3
        from config import LAMBDA_FUNCTION_NAME, AWS_REGION, AWS_PROFILE
        # profile_name sólo se usa fuera de Docker (en Docker las credenciales
        # vienen de variables de entorno, no de ~/.aws/credentials).
        _profile = AWS_PROFILE if not os.environ.get("AWS_ACCESS_KEY_ID") else None
        session = boto3.Session(region_name=AWS_REGION, profile_name=_profile)
        lam = session.client("lambda")
        payload = json.dumps({"year": year, "round": round_num, "upload": False})
        resp = lam.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType="RequestResponse",
            Payload=payload,
        )
        raw_payload = json.loads(resp["Payload"].read())
        body = raw_payload.get("body", raw_payload)
        if isinstance(body, str):
            body = json.loads(body)
        xgb_probs_dict = body.get("xgb_probs_all", {})
        logger.info("XGBoost probs obtenidas desde Lambda (%d pilotos).", len(xgb_probs_dict))
    except Exception as exc:
        logger.warning("No se pudo invocar Lambda para XGBoost probs: %s — usando modelo local.", exc)
        # Fallback: modelo local
        if MODEL_FILE.exists():
            try:
                with open(MODEL_FILE, "rb") as f:
                    xgb_model = pickle.load(f)
                X = apply_features(df_live, encoders, history_df=history_df, year=year, round_num=round_num)
                raw = xgb_model.predict_proba(X.values)[:, 1]
                # Temperature scaling (same as Lambda)
                xgb_T = 1.0
                if XGB_TEMPERATURE_FILE.exists():
                    with open(XGB_TEMPERATURE_FILE, "rb") as ft:
                        xgb_T = pickle.load(ft)
                eps = 1e-7
                raw = np.clip(raw, eps, 1 - eps)
                logits = np.log(raw / (1 - raw))
                raw = 1.0 / (1.0 + np.exp(-logits / xgb_T))
                s = raw.sum()
                normed = (raw / s if s > 0 else raw).tolist()
                xgb_probs_dict = dict(zip(drivers, normed))
            except Exception as exc2:
                logger.warning("Tampoco se pudo calcular XGBoost local para el snapshot: %s", exc2)

    snapshot = {
        "year": year,
        "round": round_num,
        "event_name": event_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drivers": drivers,
        "tab_probs": dict(zip(drivers, tab_probs.tolist())),
        "xgb_probs": xgb_probs_dict,
    }
    out = PROCESSED_DIR / f"proba_snapshot_{year}_R{round_num:02d}.json"
    out.write_text(json.dumps(snapshot, indent=2))
    logger.info("Snapshot de probabilidades guardado: %s", out)


def _print_tabnet_winner(event_name: str, winner: str, prob: float) -> None:
    """Muestra el ganador predicho por TabNet en consola."""
    print("\n" + "=" * 50)
    print(f"  PREDICCION TabNet -- {event_name.upper()}")
    print("=" * 50)
    print(f"  Ganador predicho: {winner}")
    print(f"  Probabilidad:     {prob:.1%}")
    print("=" * 50 + "\n")


# ─── Registro resultado real (lunes post-carrera) ────────────────────────────

def record_actual_result(year: int, round_num: int) -> None:
    """Actualiza el ganador real en history.csv para medir precisión."""
    import fastf1
    fastf1.Cache.enable_cache(FASTF1_CACHE)
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=False, telemetry=False, weather=False)

    results = session.results.copy()
    results["Position"] = pd.to_numeric(results["Position"], errors="coerce")

    winner = results.loc[results["Position"] == 1, "Abbreviation"].values[0]
    logger.info("Ganador real: %s", winner)

    # Obtener la posición final del piloto que cada modelo predijo,
    # para calcular MAE posicional (cuán lejos terminó del 1er puesto).
    from src.aws_utils import update_actual_winner, read_history_csv
    history = read_history_csv()
    h_mask = (history["year"] == year) & (history["round"] == round_num)

    def _finish_pos(abbr):
        if not abbr or (isinstance(abbr, float) and pd.isna(abbr)):
            return None
        row = results[results["Abbreviation"] == abbr]
        if row.empty:
            return None
        pos = row["Position"].values[0]
        return int(pos) if pd.notna(pos) else None

    xgb_finish_pos, tab_finish_pos = None, None
    if not history[h_mask].empty:
        xgb_pred = history.loc[h_mask, "predicted_winner_xgb"].values[0]
        tab_pred = history.loc[h_mask, "predicted_winner_tab"].values[0]
        xgb_finish_pos = _finish_pos(xgb_pred)
        tab_finish_pos = _finish_pos(tab_pred)
        if xgb_finish_pos is not None:
            logger.info("XGBoost predijo %s → terminó P%d", xgb_pred, xgb_finish_pos)
        if tab_finish_pos is not None:
            logger.info("TabNet  predijo %s → terminó P%d", tab_pred, tab_finish_pos)

    update_actual_winner(
        year, round_num, winner,
        xgb_finish_pos=xgb_finish_pos,
        tab_finish_pos=tab_finish_pos,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediccion local TabNet + comparativa con XGBoost (Lambda) para F1 2026"
    )
    parser.add_argument("--round",         type=int, required=True)
    parser.add_argument("--year",          type=int, default=CURRENT_SEASON)
    parser.add_argument("--no-upload",     action="store_true",
                        help="No sube resultados a S3")
    parser.add_argument("--record-result", action="store_true",
                        help="Registra el ganador real en S3 (ejecutar el lunes post-carrera)")
    args = parser.parse_args()

    if args.record_result:
        record_actual_result(args.year, args.round)
    else:
        predict_race(
            year=args.year,
            round_num=args.round,
            upload=not args.no_upload,
        )
