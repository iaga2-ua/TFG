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
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CURRENT_SEASON, ENCODER_FILE, FASTF1_CACHE, RAW_DIR,
    TABNET_MODEL_PATH, SCALER_FILE,
)
from src.data_collection import fetch_qualifying_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --- Carga de modelos --------------------------------------------------------

def load_tabnet():
    """Carga TabNet, Scaler y los encoders de etiquetas (generados durante el entrenamiento)."""
    tabnet_zip = TABNET_MODEL_PATH.with_suffix(".zip")
    if not (TABNET_MODEL_PATH.exists() or tabnet_zip.exists()) or not SCALER_FILE.exists():
        logger.warning("TabNet o Scaler no encontrados en /models. Ejecuta train.py primero.")
        return None, None, {}
    from pytorch_tabnet.tab_model import TabNetClassifier
    tab = TabNetClassifier()
    tab.load_model(str(TABNET_MODEL_PATH) + ".zip")
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    # Los encoders (circuit_encoded, constructor_encoded) son los mismos para
    # XGBoost y TabNet: generados una sola vez en build_features durante el entrenamiento.
    encoders = {}
    if ENCODER_FILE.exists():
        with open(ENCODER_FILE, "rb") as f:
            encoders = pickle.load(f)
    else:
        logger.warning("ENCODER_FILE no encontrado: circuit_encoded/constructor_encoded seran -1.")
    logger.info("TabNet + Scaler + encoders cargados.")
    return tab, scaler, encoders


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

    tab_model, scaler, encoders = load_tabnet()
    if tab_model is None:
        raise RuntimeError(
            "TabNet no disponible. Ejecuta 'python train.py --model tabnet' primero."
        )

    history_df = load_history()
    X          = apply_features(df_live, encoders, history_df=history_df, year=year, round_num=round_num)
    X_scaled   = scaler.transform(X)
    probs_tab  = tab_model.predict_proba(X_scaled)[:, 1]

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
    winner = session.results.loc[
        session.results["Position"] == 1, "Abbreviation"
    ].values[0]
    logger.info(f"🏆 Ganador real: {winner}")
    from src.aws_utils import update_actual_winner
    update_actual_winner(year, round_num, winner)


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
