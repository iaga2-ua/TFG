"""
predict_proba_all.py
====================
Muestra en terminal las probabilidades de victoria de todos los pilotos
para una ronda concreta, usando ambos modelos (XGBoost y TabNet).
No sube nada a S3.

Uso:
    python predict_proba_all.py --round 2
    python predict_proba_all.py --round 2 --year 2026
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CURRENT_SEASON,
    ENCODER_FILE,
    FASTF1_CACHE,
    MODEL_FILE,
    RAW_DIR,
    SCALER_FILE,
    TABNET_MODEL_PATH,
    TABNET_TEMPERATURE_FILE,
)
from src.data_collection import fetch_qualifying_snapshot
from src.feature_engineering import apply_features


# ─── Carga de modelos ─────────────────────────────────────────────────────────

def _load_encoders() -> dict:
    if not ENCODER_FILE.exists():
        print("  [AVISO] label_encoders.pkl no encontrado.")
        return {}
    with open(ENCODER_FILE, "rb") as f:
        return pickle.load(f)


def _load_xgboost():
    if not MODEL_FILE.exists():
        print("  [AVISO] xgboost_f1_winner.pkl no encontrado. Ejecuta train.py primero.")
        return None
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model


def _load_tabnet():
    tabnet_zip = TABNET_MODEL_PATH.with_suffix(".zip")
    if not (TABNET_MODEL_PATH.exists() or tabnet_zip.exists()):
        print("  [AVISO] tabnet_model.zip no encontrado. Ejecuta train.py primero.")
        return None, None, 1.0

    from pytorch_tabnet.tab_model import TabNetClassifier
    tab = TabNetClassifier()
    tab.load_model(str(TABNET_MODEL_PATH) + ".zip")

    if not SCALER_FILE.exists():
        print("  [AVISO] scaler.pkl no encontrado.")
        return None, None, 1.0
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    temperature = 1.0
    if TABNET_TEMPERATURE_FILE.exists():
        with open(TABNET_TEMPERATURE_FILE, "rb") as f:
            temperature = pickle.load(f)

    return tab, scaler, temperature


def _load_history() -> pd.DataFrame:
    local = RAW_DIR / "race_results_raw.csv"
    if local.exists():
        return pd.read_csv(local)
    return pd.DataFrame()


# ─── Inferencia ──────────────────────────────────────────────────────────────

def _xgboost_probs(xgb_model, X) -> np.ndarray:
    # CalibratedClassifierCV convierte el DataFrame a numpy internamente, pero
    # el modelo XGBoost tiene feature_names guardados → falla la validación.
    # Solución: borrar temporalmente los nombres antes de predecir y restaurarlos.
    boosters = [cc.estimator.get_booster() for cc in xgb_model.calibrated_classifiers_]
    saved_names = [b.feature_names for b in boosters]
    for b in boosters:
        b.feature_names = None
    try:
        data = X.values if hasattr(X, "values") else X
        probs = xgb_model.predict_proba(data)[:, 1]
    finally:
        for b, fn in zip(boosters, saved_names):
            b.feature_names = fn
    s = probs.sum()
    return probs / s if s > 0 else probs


def _tabnet_probs(tab_model, scaler, temperature: float, X: np.ndarray) -> np.ndarray:
    X_scaled  = scaler.transform(X)
    raw_probs = tab_model.predict_proba(X_scaled)[:, 1]
    eps       = 1e-7
    raw_probs = np.clip(raw_probs, eps, 1 - eps)
    logits    = np.log(raw_probs / (1 - raw_probs))
    cal_probs = 1.0 / (1.0 + np.exp(-logits / temperature))
    s         = cal_probs.sum()
    return cal_probs / s if s > 0 else cal_probs


# ─── Presentación ────────────────────────────────────────────────────────────

def _print_table(event_name: str, round_num: int, year: int, df: pd.DataFrame) -> None:
    """Imprime la tabla de probabilidades ordenada por XGBoost desc."""
    sep = "-" * 52
    header = f"  {year}  |  Ronda {round_num}  |  {event_name.upper()}"
    print()
    print("+" + "=" * 50 + "+")
    print(f"|{header:^50}|")
    print("+" + "=" * 50 + "+")
    print(f"  {'#':<3} {'PILOTO':<8} {'XGBoost':>10} {'TabNet':>10}")
    print(f"  {sep}")
    for rank, row in enumerate(df.itertuples(), start=1):
        marker = " <--" if rank == 1 else ""
        xgb_str = f"{row.xgb_prob:.1%}" if not np.isnan(row.xgb_prob) else "   N/A"
        tab_str = f"{row.tab_prob:.1%}" if not np.isnan(row.tab_prob) else "   N/A"
        print(f"  {rank:<3} {row.driver:<8} {xgb_str:>10} {tab_str:>10}{marker}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(year: int, round_num: int) -> None:
    import fastf1
    fastf1.Cache.enable_cache(FASTF1_CACHE)

    print(f"\n[INFO] Obteniendo datos de clasificación: {year} Ronda {round_num} ...")
    df_live = fetch_qualifying_snapshot(year, round_num)
    if df_live.empty:
        print("[ERROR] No se encontraron datos de clasificación.")
        sys.exit(1)

    event_name = df_live.iloc[0]["event_name"]
    drivers    = df_live["driver_abbr"].tolist()

    print("[INFO] Cargando modelos ...")
    encoders   = _load_encoders()
    history_df = _load_history()

    X = apply_features(df_live, encoders, history_df=history_df, year=year, round_num=round_num)

    # XGBoost
    xgb_probs = np.full(len(drivers), np.nan)
    xgb_model = _load_xgboost()
    if xgb_model is not None:
        xgb_probs = _xgboost_probs(xgb_model, X)

    # TabNet
    tab_probs = np.full(len(drivers), np.nan)
    tab_model, scaler, temperature = _load_tabnet()
    if tab_model is not None:
        tab_probs = _tabnet_probs(tab_model, scaler, temperature, X.values)

    # Montar tabla ordenada por XGBoost (o TabNet si XGBoost no disponible)
    sort_key = xgb_probs if not np.all(np.isnan(xgb_probs)) else tab_probs
    result_df = pd.DataFrame({
        "driver":   drivers,
        "xgb_prob": xgb_probs,
        "tab_prob": tab_probs,
    }).assign(sort_key=sort_key).sort_values("sort_key", ascending=False).drop(columns="sort_key")

    _print_table(event_name, round_num, year, result_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probabilidades de victoria (XGBoost + TabNet) sin subir a S3"
    )
    parser.add_argument("--round", type=int, required=True, help="Número de ronda")
    parser.add_argument("--year",  type=int, default=CURRENT_SEASON, help="Temporada")
    args = parser.parse_args()
    run(args.year, args.round)
    sys.exit(0)
