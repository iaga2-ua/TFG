"""
train.py  (raíz del proyecto)
==============================
Entrena el ecosistema dual de modelos para F1.
Ambos modelos se entrenan con los MISMOS datos históricos (FastF1).

Despliegue:
  - XGBoost  → artefactos subidos a S3, inferencia vía AWS Lambda.
  - TabNet   → artefactos guardados en /models local, inferencia vía predict.py.

Uso:
    python train.py                        # entrena ambos modelos
    python train.py --model xgboost        # solo XGBoost
    python train.py --model tabnet         # solo TabNet
    python train.py --upload-s3            # sube artefactos XGBoost a S3 al final
    python train.py --model xgboost --upload-s3
    python train.py --optimize             # optimiza HPs con Optuna, luego entrena ambos
    python train.py --optimize --n-trials 50   # 50 trials de Optuna
    python train.py --model tabnet --optimize  # solo TabNet, con optimización
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# La raíz del proyecto ya está en sys.path cuando se ejecuta desde aquí,
# pero lo aseguramos explícitamente para llamadas indirectas.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    ENCODER_FILE, FEATURE_COLS, MODEL_FILE, OPTUNA_N_TRIALS,
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


# ─── Optimización de hiperparámetros (Optuna) ────────────────────────────────

def optimize_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int) -> dict:
    """Busca los mejores hiperparámetros para XGBoost con Optuna (ROC-AUC, 5-fold CV)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {**XGBOOST_PARAMS, **{
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        }}
        clf = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"   Mejor ROC-AUC XGBoost (Optuna): {study.best_value:.4f}")
    logger.info(f"   Mejores params XGBoost: {study.best_params}")
    # Combinar parámetros fijos con los optimizados
    return {**XGBOOST_PARAMS, **study.best_params}


def optimize_tabnet(X_scaled: np.ndarray, y: np.ndarray, n_trials: int) -> dict:
    """Busca los mejores hiperparámetros para TabNet con Optuna (ROC-AUC, val 20%)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    def objective(trial: optuna.Trial) -> float:
        n_d = trial.suggest_int("n_d", 8, 64)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        clf = TabNetClassifier(
            n_d=n_d,
            n_a=n_d,
            n_steps=trial.suggest_int("n_steps", 3, 8),
            gamma=trial.suggest_float("gamma", 1.0, 2.0),
            lambda_sparse=trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=trial.suggest_float("lr", 1e-3, 5e-2, log=True)),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            mask_type="entmax",
            verbose=0,
        )
        clf.fit(
            X_train=X_tr, y_train=y_tr,
            eval_set=[(X_val, y_val)],
            max_epochs=50,
            patience=10,
            batch_size=batch_size,
            virtual_batch_size=min(batch_size // 4, 128),
            num_workers=0,
            drop_last=False,
        )
        proba = clf.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, proba))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"   Mejor ROC-AUC TabNet (Optuna): {study.best_value:.4f}")
    logger.info(f"   Mejores params TabNet: {study.best_params}")
    return study.best_params


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train(upload_s3: bool = False, model: str = "all", optimize: bool = False, n_trials: int = OPTUNA_N_TRIALS) -> None:
    # 1. Datos y features (siempre los mismos para ambos modelos)
    df_raw = load_or_fetch_raw()
    X, y, encoders = build_features(df_raw)

    xgb_model = None
    tab_model  = None

    # 2. XGBoost — inferencia en AWS Lambda
    if model in ("xgboost", "all"):
        if optimize:
            logger.info(f"🔍 Optimizando XGBoost ({n_trials} trials de Optuna)...")
            xgb_params = optimize_xgboost(X.values, y.values, n_trials)
        else:
            xgb_params = XGBOOST_PARAMS
        logger.info(f"🌲 Entrenando XGBoost ({len(X)} filas, {X.shape[1]} features)...")
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(X, y)
        logger.info("   XGBoost listo.")

    # 3. TabNet — inferencia solo en local
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model in ("tabnet", "all"):
        if optimize:
            logger.info(f"🔍 Optimizando TabNet ({n_trials} trials de Optuna)...")
            best_tab  = optimize_tabnet(X_scaled, y.values, n_trials)
            tab_lr    = best_tab.pop("lr")
            tab_batch = best_tab.pop("batch_size")
            tab_model = TabNetClassifier(
                n_a=best_tab["n_d"],
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=tab_lr),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"step_size": 10, "gamma": 0.9},
                mask_type="entmax",
                **best_tab,
            )
        else:
            tab_batch = 1024
            tab_model = TabNetClassifier(
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"step_size": 10, "gamma": 0.9},
                mask_type="entmax",
            )
        logger.info("🧠 Entrenando TabNet (inferencia local)...")
        tab_model.fit(
            X_train=X_scaled,
            y_train=y.values,
            max_epochs=50,
            patience=10,
            batch_size=tab_batch,
            virtual_batch_size=min(tab_batch // 4, 128),
            num_workers=0,
            drop_last=False,
        )
        logger.info("   TabNet listo.")

    # 4. Métricas de importancia → Looker Studio
    if xgb_model is not None and tab_model is not None:
        logger.info("📊 Calculando importancia de variables (ambos modelos)...")
        importance_df = pd.DataFrame({
            "feature":             FEATURE_COLS,
            "importance_xgboost":  xgb_model.feature_importances_,
            "importance_tabnet":   tab_model.feature_importances_,
        }).sort_values("importance_xgboost", ascending=False)
    elif xgb_model is not None:
        logger.info("📊 Calculando importancia de variables (XGBoost)...")
        importance_df = pd.DataFrame({
            "feature":             FEATURE_COLS,
            "importance_xgboost":  xgb_model.feature_importances_,
        }).sort_values("importance_xgboost", ascending=False)
    elif tab_model is not None:
        logger.info("📊 Calculando importancia de variables (TabNet)...")
        importance_df = pd.DataFrame({
            "feature":             FEATURE_COLS,
            "importance_tabnet":   tab_model.feature_importances_,
        })
    else:
        importance_df = pd.DataFrame()

    if not importance_df.empty:
        # trained_at permite filtrar por version del modelo en Looker Studio
        from datetime import datetime, timezone
        importance_df["trained_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not importance_df.empty:
        feat_path = PROCESSED_DIR / "feature_importance.csv"
        importance_df.to_csv(feat_path, index=False)
        logger.info(f"   Importancia guardada → {feat_path}")

    # Comparativa historica (solo si se entrenaron ambos)
    if xgb_model is not None and tab_model is not None:
        # Agregar a nivel de carrera: ganador predicho vs real
        tmp = pd.DataFrame({
            "driver":        df_raw.loc[X.index, "driver_abbr"].values,
            "season":        df_raw.loc[X.index, "year"].values,
            "round":         df_raw.loc[X.index, "round"].values,
            "actual_winner": y.values,
            "prob_xgboost":  xgb_model.predict_proba(X)[:, 1],
            "prob_tabnet":   tab_model.predict_proba(X_scaled)[:, 1],
        })
        race_rows = []
        for (season, rnd), grp in tmp.groupby(["season", "round"]):
            xgb_winner = grp.loc[grp["prob_xgboost"].idxmax(), "driver"]
            tab_winner = grp.loc[grp["prob_tabnet"].idxmax(), "driver"]
            actual     = grp.loc[grp["actual_winner"] == 1, "driver"]
            actual_abbr = actual.values[0] if len(actual) > 0 else None
            race_rows.append({
                "season":               int(season),
                "round":                int(rnd),
                "predicted_winner_xgb": xgb_winner,
                "predicted_winner_tab": tab_winner,
                "actual_winner":        actual_abbr,
                "xgb_correct":          int(xgb_winner == actual_abbr) if actual_abbr else None,
                "tab_correct":          int(tab_winner == actual_abbr) if actual_abbr else None,
            })
        comparison_df = pd.DataFrame(race_rows).sort_values(["season", "round"])
        comp_path = PROCESSED_DIR / "historical_model_performance.csv"
        comparison_df.to_csv(comp_path, index=False)
        logger.info(f"   Comparativa historica guardada -> {comp_path}")

    # 5. Guardar artefactos locales
    logger.info("💾 Guardando artefactos en /models ...")
    if xgb_model is not None:
        with open(MODEL_FILE,   "wb") as f: pickle.dump(xgb_model, f)
        with open(ENCODER_FILE, "wb") as f: pickle.dump(encoders,  f)
        logger.info("   ✅ XGBoost + encoders guardados.")
    if tab_model is not None:
        with open(SCALER_FILE, "wb") as f: pickle.dump(scaler, f)
        tab_model.save_model(str(TABNET_FILE))   # genera tabnet_model.zip
        logger.info("   ✅ TabNet + Scaler guardados (solo local).")

    # 6. Subir artefactos XGBoost a S3 → disponible para AWS Lambda
    if upload_s3:
        if xgb_model is None:
            logger.warning("⚠️  --upload-s3 ignorado: XGBoost no fue entrenado en esta ejecución.")
        else:
            try:
                from src.aws_utils import upload_model_artefacts, upload_to_s3
                upload_model_artefacts()   # sube MODEL_FILE + ENCODER_FILE
                if not importance_df.empty:
                    upload_to_s3(feat_path,  "metrics/feature_importance.csv")
                if xgb_model is not None and tab_model is not None:
                    upload_to_s3(comp_path, "metrics/historical_performance.csv")
                logger.info("🚀 Artefactos XGBoost subidos a S3 → listos para Lambda.")
            except Exception as exc:
                logger.error(f"❌ Error al subir a S3: {exc}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento XGBoost (Lambda) + TabNet (local) para F1 2026"
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "tabnet", "all"],
        default="all",
        help="Modelo a entrenar. 'all' entrena ambos con los mismos datos (por defecto).",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Sube artefactos XGBoost a S3 tras el entrenamiento (para despliegue Lambda).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimiza hiperparámetros con Optuna antes de entrenar (incrementa el tiempo).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=OPTUNA_N_TRIALS,
        metavar="N",
        help=f"Número de trials de Optuna (por defecto: {OPTUNA_N_TRIALS}).",
    )
    args = parser.parse_args()
    train(upload_s3=args.upload_s3, model=args.model, optimize=args.optimize, n_trials=args.n_trials)
