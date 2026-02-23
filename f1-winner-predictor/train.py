"""
train.py  (raíz del proyecto)
==============================
Entrena el ecosistema dual de modelos para F1:
  1. XGBoost  – modelo robusto basado en árboles (desplegado en S3/Lambda).
  2. TabNet   – red neuronal de atención (uso local).

Genera métricas de importancia de variables para Looker Studio.

Uso:
    python train.py               # entrena y guarda en /models y /data/processed
    python train.py --upload-s3   # además sube artefactos XGBoost + métricas a S3
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# La raíz del proyecto ya está en sys.path cuando se ejecuta desde aquí,
# pero lo aseguramos explícitamente para llamadas indirectas.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    ENCODER_FILE, FEATURE_COLS, MODEL_FILE,
    PROCESSED_DIR, RAW_DIR, TRAIN_SEASONS, XGBOOST_PARAMS,
)
from src.data_collection import collect_all_seasons
from src.feature_engineering import build_features

# ─── Rutas de artefactos locales ─────────────────────────────────────────────
TABNET_FILE = ROOT / "models" / "tabnet_model"   # pytorch-tabnet añade .zip
SCALER_FILE = ROOT / "models" / "scaler.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Carga de datos ───────────────────────────────────────────────────────────

def load_or_fetch_raw() -> pd.DataFrame:
    """Carga el CSV histórico local; si no existe, lo descarga de FastF1."""
    raw_path = RAW_DIR / "race_results_raw.csv"
    if raw_path.exists():
        logger.info(f"📂 Cargando datos crudos desde {raw_path}")
        return pd.read_csv(raw_path)
    logger.info("📡 Datos locales no encontrados — descargando de FastF1...")
    return collect_all_seasons(seasons=TRAIN_SEASONS, save=True)


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train(upload_s3: bool = False) -> None:
    # 1. Datos y features
    df_raw = load_or_fetch_raw()
    X, y, encoders = build_features(df_raw)

    # TabNet requiere normalización Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. XGBoost
    logger.info(f"🌲 Entrenando XGBoost con {len(X)} filas y {X.shape[1]} features...")
    xgb_model = XGBClassifier(**XGBOOST_PARAMS)
    xgb_model.fit(X, y)
    logger.info("   XGBoost listo.")

    # 3. TabNet
    logger.info("🧠 Entrenando TabNet (Deep Learning)...")
    tab_model = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type="entmax",
    )
    tab_model.fit(
        X_train=X_scaled,
        y_train=y.values,
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    logger.info("   TabNet listo.")

    # 4. Métricas de importancia → Looker Studio
    logger.info("📊 Calculando importancia de variables...")
    importance_df = pd.DataFrame({
        "feature":             FEATURE_COLS,
        "importance_xgboost":  xgb_model.feature_importances_,
        "importance_tabnet":   tab_model.feature_importances_,
    }).sort_values("importance_xgboost", ascending=False)

    feat_path = PROCESSED_DIR / "feature_importance.csv"
    importance_df.to_csv(feat_path, index=False)
    logger.info(f"   Importancia guardada → {feat_path}")

    # Comparativa histórica de los dos modelos
    comparison_df = pd.DataFrame({
        "driver":         df_raw.loc[X.index, "driver_abbr"].values,
        "season":         df_raw.loc[X.index, "year"].values,
        "round":          df_raw.loc[X.index, "round"].values,
        "actual_winner":  y.values,
        "prob_xgboost":   xgb_model.predict_proba(X)[:, 1],
        "prob_tabnet":    tab_model.predict_proba(X_scaled)[:, 1],
    })
    comp_path = PROCESSED_DIR / "historical_model_performance.csv"
    comparison_df.to_csv(comp_path, index=False)
    logger.info(f"   Comparativa histórica guardada → {comp_path}")

    # 5. Guardar artefactos
    logger.info("💾 Guardando modelos y transformadores en /models ...")
    with open(MODEL_FILE, "wb")   as f: pickle.dump(xgb_model, f)
    with open(ENCODER_FILE, "wb") as f: pickle.dump(encoders, f)
    with open(SCALER_FILE, "wb")  as f: pickle.dump(scaler, f)
    tab_model.save_model(str(TABNET_FILE))          # genera tabnet_model.zip
    logger.info("✅ Todos los modelos guardados en /models.")

    # 6. Sincronización con AWS (opcional)
    if upload_s3:
        try:
            from src.aws_utils import upload_model_artefacts, upload_to_s3
            upload_model_artefacts()                           # XGBoost + encoders
            upload_to_s3(feat_path,  "metrics/feature_importance.csv")
            upload_to_s3(comp_path,  "metrics/historical_performance.csv")
            logger.info("🚀 Artefactos y métricas subidos a S3.")
        except Exception as exc:
            logger.error(f"❌ Error al subir a S3: {exc}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento dual XGBoost + TabNet para F1 2026")
    parser.add_argument("--upload-s3", action="store_true",
                        help="Subir artefactos XGBoost y métricas a S3 tras el entrenamiento")
    args = parser.parse_args()
    train(upload_s3=args.upload_s3)
