"""
predict.py  (raíz del proyecto)
================================
Script de inferencia post-clasificación (Sábado).
Genera probabilidades de victoria comparando XGBoost y TabNet para la parrilla 2026.

Uso:
    python predict.py --round 1                   # predice ronda 1, sube a S3
    python predict.py --round 5 --no-upload       # no sube a S3
    python predict.py --round 5 --from-s3         # descarga modelo de S3 primero
    python predict.py --round 5 --record-result   # registra ganador real (post-carrera)
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
    CURRENT_SEASON, ENCODER_FILE, FASTF1_CACHE, MODEL_FILE, RAW_DIR,
)
from src.data_collection import fetch_qualifying_snapshot
from src.feature_engineering import apply_features

# Rutas artefactos locales
TABNET_MODEL_PATH = ROOT / "models" / "tabnet_model.zip"
SCALER_FILE       = ROOT / "models" / "scaler.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Carga de modelos ─────────────────────────────────────────────────────────

def load_xgboost(from_s3: bool = False):
    """Carga XGBoost y encoders. Descarga de S3 si se indica."""
    if not MODEL_FILE.exists() or from_s3:
        from src.aws_utils import download_model_artefacts
        download_model_artefacts()
    with open(MODEL_FILE,   "rb") as f: xgb = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f: enc = pickle.load(f)
    logger.info("✅ XGBoost + encoders cargados.")
    return xgb, enc


def load_tabnet():
    """Carga TabNet y Scaler (solo disponibles en entorno local)."""
    if not TABNET_MODEL_PATH.exists() or not SCALER_FILE.exists():
        logger.warning("⚠️  TabNet o Scaler no encontrados en /models — se omitirá.")
        return None, None
    from pytorch_tabnet.tab_model import TabNetClassifier
    tab = TabNetClassifier()
    tab.load_model(str(TABNET_MODEL_PATH))
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    logger.info("✅ TabNet + Scaler cargados.")
    return tab, scaler


# ─── Carga de historial ───────────────────────────────────────────────────────

def load_history() -> pd.DataFrame:
    """
    Intenta leer history.csv desde S3.
    Si falla, usa el CSV local de datos históricos como fallback.
    """
    try:
        from src.aws_utils import read_history_csv
        df = read_history_csv()
        logger.info(f"📥 Historial cargado desde S3: {len(df)} filas.")
        return df
    except Exception as exc:
        logger.warning(f"S3 no disponible ({exc}). Usando datos locales.")
    local = RAW_DIR / "race_results_raw.csv"
    if local.exists():
        return pd.read_csv(local)
    logger.warning("Sin historial disponible — las rolling features serán NaN.")
    return pd.DataFrame()


# ─── Núcleo de predicción ────────────────────────────────────────────────────

def predict_race(year: int, round_num: int,
                 upload: bool = True, from_s3: bool = False) -> pd.DataFrame:
    logger.info(f"🏎️  Iniciando predicción: {year} Round {round_num}")

    # 1. Datos de parrilla post-clasificación (22 pilotos)
    df_live = fetch_qualifying_snapshot(year, round_num)

    # 2. Modelos
    xgb_model, encoders = load_xgboost(from_s3=from_s3)
    tab_model, scaler   = load_tabnet()

    # 3. Features de inferencia (con historial para rolling features)
    history_df = load_history()
    X = apply_features(df_live, encoders, history_df=history_df)

    # 4. Inferencia dual
    probs_xgb = xgb_model.predict_proba(X)[:, 1]

    probs_tab = np.zeros(len(X))
    if tab_model is not None:
        X_scaled  = scaler.transform(X)
        probs_tab = tab_model.predict_proba(X_scaled)[:, 1]

    # 5. DataFrame de resultados
    results = df_live[[
        "year", "round", "event_name", "circuit",
        "driver_abbr", "full_name", "team", "grid_position",
    ]].copy()

    results["win_prob_xgboost"]       = probs_xgb
    results["win_prob_tabnet"]        = probs_tab
    results["prediction_timestamp"]   = datetime.now(timezone.utc).isoformat()
    results["predicted_winner_xgb"]   = (probs_xgb == probs_xgb.max()).astype(int)
    results["predicted_winner_tab"]   = (probs_tab == probs_tab.max()).astype(int) if tab_model else 0

    # Ordenar por XGBoost (modelo principal)
    results = results.sort_values("win_prob_xgboost", ascending=False).reset_index(drop=True)
    results["predicted_rank"] = results.index + 1

    # 6. Consola
    _print_dual_comparison(results)

    # 7. Subir a S3
    if upload:
        try:
            from src.aws_utils import append_to_history_csv
            append_to_history_csv(results)
            logger.info("☁️  Predicciones sincronizadas con S3.")
        except Exception as exc:
            logger.error(f"❌ Error al subir a S3: {exc}")
            local_out = RAW_DIR / f"prediction_{year}_R{round_num:02d}.csv"
            results.to_csv(local_out, index=False)
            logger.info(f"💾 Guardado localmente → {local_out}")

    return results


def _print_dual_comparison(df: pd.DataFrame) -> None:
    """Tabla comparativa XGBoost vs TabNet en consola."""
    print("\n" + "═" * 85)
    print(f"  🏁  PREDICCIÓN 2026 — {df['event_name'].iloc[0].upper()}")
    print("═" * 85)
    print(f"  {'Driver':<12} {'Grid':<6} {'XGBoost %':<15} {'TabNet %':<15} {'Diferencia'}")
    print("─" * 85)
    for _, row in df.head(10).iterrows():
        diff = row["win_prob_xgboost"] - row["win_prob_tabnet"]
        diff_str = f"{diff:>+7.1%}" if row["win_prob_tabnet"] > 0 else "    N/A"
        xgb_mark = "🏆" if row["predicted_winner_xgb"] else "  "
        print(
            f"  {xgb_mark}{row['driver_abbr']:<11} {int(row['grid_position']):<6}"
            f" {row['win_prob_xgboost']:>10.1%}      {row['win_prob_tabnet']:>10.1%}"
            f"      {diff_str}"
        )
    print("═" * 85 + "\n")


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
    parser = argparse.ArgumentParser(description="Predicción ganador F1 2026 (post-clasificación)")
    parser.add_argument("--round",          type=int, required=True)
    parser.add_argument("--year",           type=int, default=CURRENT_SEASON)
    parser.add_argument("--no-upload",      action="store_true", help="No sube a S3")
    parser.add_argument("--from-s3",        action="store_true", help="Descarga modelo de S3")
    parser.add_argument("--record-result",  action="store_true", help="Registra ganador real (post-carrera)")
    args = parser.parse_args()

    if args.record_result:
        record_actual_result(args.year, args.round)
    else:
        predict_race(
            year=args.year,
            round_num=args.round,
            upload=not args.no_upload,
            from_s3=args.from_s3,
        )
