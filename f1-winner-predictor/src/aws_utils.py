"""
aws_utils.py
============
Gestión de S3 y Athena para el Predictor F1 2026.
Soporta comparación dual: XGBoost (AWS) vs TabNet (Local).
"""

import io
import logging
import os
from pathlib import Path
import boto3
import pandas as pd

# Importar configuraciones
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    AWS_REGION, AWS_PROFILE,
    S3_BUCKET, S3_HISTORY_KEY, S3_MODEL_KEY, S3_ENCODER_KEY, S3_RACE_RESULTS_KEY,
    MODEL_FILE, ENCODER_FILE, RAW_DIR,
    ATHENA_DATABASE, ATHENA_TABLE, ATHENA_TABLE_IMPORTANCE,
    ATHENA_OUTPUT_LOC,
)

logger = logging.getLogger(__name__)

# --- Columnas del historial (UNA fila por carrera) ---
# Lambda escribe: predicted_winner_xgb + win_prob_xgboost
# Local  escribe: predicted_winner_tab + win_prob_tabnet
# Post-carrera:   actual_winner + xgb_correct + tab_correct (via update_actual_winner)
HISTORY_COLUMNS = [
    "year", "round", "event_name", "circuit",
    "predicted_winner_xgb", "win_prob_xgboost",
    "predicted_winner_tab", "win_prob_tabnet",
    "actual_winner",
    "xgb_correct", "tab_correct",
    "prediction_timestamp",
]

# ─── GESTIÓN DE SESIONES ──────────────────────────────────────────────────────

def _get_session() -> boto3.Session:
    """Detecta automáticamente el entorno (Local vs Lambda)."""
    profile = os.environ.get("AWS_PROFILE", AWS_PROFILE)
    try:
        # Intenta usar perfil local
        return boto3.Session(region_name=AWS_REGION, profile_name=profile)
    except Exception:
        # Fallback para Lambda (IAM Role)
        return boto3.Session(region_name=AWS_REGION)

def _s3_client():
    return _get_session().client("s3")

# ─── GESTIÓN DEL HISTÓRICO (S3) ───────────────────────────────────────────────

def read_history_csv() -> pd.DataFrame:
    """Descarga el historial de S3 o crea uno nuevo."""
    s3 = _s3_client()
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_HISTORY_KEY)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        logger.info("Creando nuevo archivo histórico en S3...")
        return pd.DataFrame(columns=HISTORY_COLUMNS)

def append_to_history_csv(new_row) -> None:
    """
    Upsert de UNA fila (una por carrera) en predictions/history.csv.

    new_row puede ser:
      - dict:           {"year": 2026, "round": 5, "predicted_winner_xgb": "VER", ...}
      - DataFrame:      una sola fila con las mismas columnas

    Estrategia:
      - Lambda escribe predicted_winner_xgb (predicted_winner_tab = NaN).
      - Local  escribe predicted_winner_tab (predicted_winner_xgb = NaN).
      - update(overwrite=False) fusiona ambas escrituras en la misma fila.
    """
    import numpy as np

    if isinstance(new_row, dict):
        new_row = pd.DataFrame([new_row])
    else:
        new_row = new_row.copy()

    existing = read_history_csv()

    for col in HISTORY_COLUMNS:
        if col not in new_row.columns:
            new_row[col] = np.nan
        if col not in existing.columns:
            existing[col] = np.nan

    key_cols  = ["year", "round"]
    new_idx   = new_row.set_index(key_cols)

    if existing.empty:
        combined = new_row[HISTORY_COLUMNS]
    else:
        existing_idx = existing.set_index(key_cols)
        # Sobrescribir solo celdas no-NaN del new_row: permite correcciones
        # pero preserva columnas que la otra parte (XGB vs TabNet) ya escribió.
        # Solo actualizamos filas que YA existen en existing_idx; las nuevas
        # se añaden vía new_only más abajo (evita KeyError en MultiIndex).
        for col in new_idx.columns:
            non_nan = new_idx[col].notna()
            overlap = new_idx.index[non_nan].intersection(existing_idx.index)
            if not overlap.empty:
                existing_idx.loc[overlap, col] = new_idx.loc[overlap, col]
        new_only = new_idx[~new_idx.index.isin(existing_idx.index)]
        combined = pd.concat([existing_idx, new_only]).reset_index()
        combined = combined[[c for c in HISTORY_COLUMNS if c in combined.columns]]

    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    _s3_client().put_object(
        Bucket=S3_BUCKET,
        Key=S3_HISTORY_KEY,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    logger.info("history.csv actualizado en s3://%s/%s", S3_BUCKET, S3_HISTORY_KEY)

# ─── SUBIDA DE ARCHIVOS GENÉRICA ─────────────────────────────────────────────

def upload_to_s3(local_path: Path, s3_key: str) -> None:
    """Sube cualquier archivo (como las métricas de importancia) a S3."""
    try:
        _s3_client().upload_file(str(local_path), S3_BUCKET, s3_key)
        logger.info(f"⬆️ Archivo {local_path.name} subido a {s3_key}")
    except Exception as e:
        logger.error(f"❌ Error subiendo a S3: {e}")

# ─── MODELOS Y ARTEFACTOS ────────────────────────────────────────────────────

def upload_model_artefacts() -> None:
    """Sube XGBoost, Encoders y el CSV de resultados históricos de carrera."""
    artefacts = [
        (MODEL_FILE, S3_MODEL_KEY),
        (ENCODER_FILE, S3_ENCODER_KEY),
        (RAW_DIR / "race_results_raw.csv", S3_RACE_RESULTS_KEY),
    ]
    for local, key in artefacts:
        if local.exists():
            upload_to_s3(local, key)

def download_model_artefacts() -> None:
    """Descarga artefactos (usado por Lambda)."""
    s3 = _s3_client()
    for local, key in [(MODEL_FILE, S3_MODEL_KEY), (ENCODER_FILE, S3_ENCODER_KEY)]:
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local))

def download_race_results() -> pd.DataFrame:
    """Descarga race_results_raw.csv de S3 (usado por Lambda para features históricas)."""
    s3 = _s3_client()
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_RACE_RESULTS_KEY)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as exc:
        logger.warning("No se pudo descargar race_results_raw.csv: %s", exc)
        return pd.DataFrame()


# ─── ATHENA (MOTOR PARA LOOKER STUDIO) ───────────────────────────────────────

def setup_athena_table() -> None:
    """
    Crea (o verifica) las dos tablas externas de Athena sobre S3:
      1. race_predictions   -- un ganador predicho por carrera (XGBoost + TabNet)
      2. feature_importance -- importancia de cada feature con trained_at
    """
    ddl_predictions = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DATABASE}.{ATHENA_TABLE} (
        year INT,
        round INT,
        event_name STRING,
        circuit STRING,
        predicted_winner_xgb STRING,
        win_prob_xgboost DOUBLE,
        predicted_winner_tab STRING,
        win_prob_tabnet DOUBLE,
        actual_winner STRING,
        xgb_correct INT,
        tab_correct INT,
        prediction_timestamp STRING
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LOCATION 's3://{S3_BUCKET}/predictions/'
    TBLPROPERTIES ('skip.header.line.count'='1');
    """

    ddl_importance = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DATABASE}.{ATHENA_TABLE_IMPORTANCE} (
        feature STRING,
        importance_xgboost DOUBLE,
        importance_tabnet DOUBLE,
        trained_at STRING
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LOCATION 's3://{S3_BUCKET}/metrics/'
    TBLPROPERTIES (
        'skip.header.line.count'='1',
        'classification.file.pattern'='feature_importance.csv'
    );
    """

    for name, ddl in [
        (ATHENA_TABLE,            ddl_predictions),
        (ATHENA_TABLE_IMPORTANCE, ddl_importance),
    ]:
        _run_athena_query(ddl)
        logger.info("Tabla Athena '%s' verificada/creada.", name)


def _run_athena_query(query: str):
    client = _get_session().client("athena")
    return client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_LOC}
    )

def update_actual_winner(year: int, round_num: int, winner_abbr: str):
    """
    Registra el ganador real tras la carrera y calcula xgb_correct / tab_correct.
    Llamar el lunes post-carrera via: python predict.py --round N --record-result
    """
    df = read_history_csv()
    mask = (df["year"] == year) & (df["round"] == round_num)

    if df[mask].empty:
        logger.warning("No hay prediccion en S3 para %d R%d.", year, round_num)
        return

    df.loc[mask, "actual_winner"] = winner_abbr
    df.loc[mask, "xgb_correct"]   = (
        df.loc[mask, "predicted_winner_xgb"] == winner_abbr
    ).astype(int)
    df.loc[mask, "tab_correct"]   = (
        df.loc[mask, "predicted_winner_tab"] == winner_abbr
    ).astype(int)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    _s3_client().put_object(
        Bucket=S3_BUCKET,
        Key=S3_HISTORY_KEY,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    logger.info(
        "Ganador real '%s' registrado para %d R%d. XGB correcto: %s | TAB correcto: %s",
        winner_abbr, year, round_num,
        bool(df.loc[mask, "xgb_correct"].values[0]),
        bool(df.loc[mask, "tab_correct"].values[0]),
    )
    # Actualizar la gráfica de accuracy en Google Sheets
    try:
        sync_to_sheets()
        sync_model_accuracy_to_sheets()
    except Exception as exc:
        logger.warning("No se pudo sincronizar accuracy a Sheets: %s", exc)

# ─── SINCRONIZACIÓN CON GOOGLE SHEETS ────────────────────────────────────────

SHEETS_ID = "1Jw7wo3bqC2IS9MmfSJe6T2waQyp7LTCMv4al7gwhtPI"

def sync_to_sheets(credentials_path: str = None) -> None:
    """
    Sincroniza history.csv de S3 con la Google Sheet del TFG.

    credentials_path: ruta al JSON de la service account de Google.
    Si es None, usa la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
    o el archivo .google_credentials.json en la raíz del proyecto.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    # Resolver credenciales
    if credentials_path is None:
        credentials_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            str(Path(__file__).parent.parent / ".google_credentials.json"),
        )

    if not Path(credentials_path).exists():
        logger.error(
            "Credenciales de Google no encontradas en %s. "
            "Descarga el JSON de la service account y colócalo ahí.",
            credentials_path,
        )
        return

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    client = gspread.authorize(creds)

    # Leer datos de S3
    df = read_history_csv()
    if df.empty:
        logger.warning("history.csv vacío — nada que sincronizar.")
        return

    df = df.fillna("")

    # Escribir en la hoja
    sheet     = client.open_by_key(SHEETS_ID)
    worksheet = sheet.get_worksheet(0)
    worksheet.clear()
    worksheet.update([df.columns.tolist()] + df.values.tolist())

    logger.info("✅ Google Sheets sincronizado: %d filas.", len(df))


def sync_feature_importance_to_sheets(credentials_path: str = None) -> None:
    """
    Sincroniza metrics/feature_importance.csv de S3 con la pestaña
    'feature_importance' de la misma Google Sheet del TFG.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    if credentials_path is None:
        credentials_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            str(Path(__file__).parent.parent / ".google_credentials.json"),
        )

    if not Path(credentials_path).exists():
        logger.error(
            "Credenciales de Google no encontradas en %s.",
            credentials_path,
        )
        return

    # Leer feature_importance.csv de S3
    s3 = _s3_client()
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="metrics/feature_importance.csv")
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        logger.warning("metrics/feature_importance.csv no existe en S3 — entrena primero los modelos.")
        return

    df = df.fillna("")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(SHEETS_ID)

    # Obtener o crear la pestaña 'feature_importance'
    try:
        ws = sheet.worksheet("feature_importance")
    except gspread.exceptions.WorksheetNotFound:
        ws = sheet.add_worksheet(title="feature_importance", rows=50, cols=10)

    ws.clear()
    ws.update([df.columns.tolist()] + df.values.tolist())

    logger.info("✅ feature_importance sincronizado a Google Sheets: %d features.", len(df))


def sync_model_accuracy_to_sheets(credentials_path: str = None) -> None:
    """
    Sincroniza metrics/historical_performance.csv de S3 con la pestaña
    'model_accuracy' de la misma Google Sheet del TFG.

    Añade columnas de accuracy acumulada para ver la evolución por carrera.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    if credentials_path is None:
        credentials_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            str(Path(__file__).parent.parent / ".google_credentials.json"),
        )

    if not Path(credentials_path).exists():
        logger.error(
            "Credenciales de Google no encontradas en %s.",
            credentials_path,
        )
        return

    # Leer predictions/history.csv — predicciones reales de 2026
    df = read_history_csv()
    if df.empty:
        logger.warning("history.csv vacío — aún no hay predicciones de 2026.")
        return

    # Ordenar cronológicamente y calcular accuracy acumulada solo sobre carreras ya disputadas
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    df["race_label"] = df["event_name"].fillna(df["year"].astype(str) + " R" + df["round"].astype(str))
    df["xgb_correct"] = pd.to_numeric(df["xgb_correct"], errors="coerce")
    df["tab_correct"] = pd.to_numeric(df["tab_correct"], errors="coerce")
    valid = df["xgb_correct"].notna()
    df.loc[valid, "xgb_accuracy_cumul"] = df.loc[valid, "xgb_correct"].expanding().mean().round(4)
    df.loc[valid, "tab_accuracy_cumul"] = df.loc[valid, "tab_correct"].expanding().mean().round(4)

    df = df.fillna("")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(SHEETS_ID)

    try:
        ws = sheet.worksheet("model_accuracy")
    except gspread.exceptions.WorksheetNotFound:
        ws = sheet.add_worksheet(title="model_accuracy", rows=200, cols=15)

    ws.clear()
    ws.update([df.columns.tolist()] + df.values.tolist())

    logger.info("✅ model_accuracy sincronizado a Google Sheets: %d carreras.", len(df))