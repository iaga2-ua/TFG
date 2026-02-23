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
    S3_BUCKET, S3_HISTORY_KEY, S3_MODEL_KEY, S3_ENCODER_KEY,
    MODEL_FILE, ENCODER_FILE,
    ATHENA_DATABASE, ATHENA_TABLE, ATHENA_OUTPUT_LOC,
)

logger = logging.getLogger(__name__)

# --- Columnas finales para Looker Studio ---
HISTORY_COLUMNS = [
    "year", "round", "event_name", "circuit",
    "driver_abbr", "team", "grid_position",
    "win_prob_xgboost", 
    "win_prob_tabnet", 
    "actual_winner", 
    "prediction_timestamp"
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

def append_to_history_csv(new_rows: pd.DataFrame) -> None:
    """Combina predicciones y evita duplicados de los 22 pilotos."""
    existing = read_history_csv()

    # Asegurar consistencia de columnas
    for col in HISTORY_COLUMNS:
        if col not in new_rows.columns: new_rows[col] = 0
        if col not in existing.columns: existing[col] = 0

    combined = pd.concat([existing, new_rows[HISTORY_COLUMNS]], ignore_index=True)
    
    # Sincronización: El resultado local (TabNet) actualiza el de la nube (XGB)
    combined = combined.drop_duplicates(
        subset=["year", "round", "driver_abbr"], keep="last"
    )
    
    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    _s3_client().put_object(
        Bucket=S3_BUCKET,
        Key=S3_HISTORY_KEY,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv"
    )
    logger.info(f"✅ Historial sincronizado en s3://{S3_BUCKET}/{S3_HISTORY_KEY}")

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
    """Sube XGBoost y Encoders."""
    for local, key in [(MODEL_FILE, S3_MODEL_KEY), (ENCODER_FILE, S3_ENCODER_KEY)]:
        if local.exists():
            upload_to_s3(local, key)

def download_model_artefacts() -> None:
    """Descarga artefactos (usado por Lambda)."""
    s3 = _s3_client()
    for local, key in [(MODEL_FILE, S3_MODEL_KEY), (ENCODER_FILE, S3_ENCODER_KEY)]:
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local))

# ─── ATHENA (MOTOR PARA LOOKER STUDIO) ───────────────────────────────────────

def setup_athena_table() -> None:
    """Configura la tabla para que Looker Studio 'vea' los datos."""
    # Nota: Usamos STRING para el timestamp para evitar problemas de formato en Athena
    ddl = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DATABASE}.{ATHENA_TABLE} (
        year INT,
        round INT,
        event_name STRING,
        circuit STRING,
        driver_abbr STRING,
        team STRING,
        grid_position INT,
        win_prob_xgboost DOUBLE,
        win_prob_tabnet DOUBLE,
        actual_winner INT,
        prediction_timestamp STRING
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LOCATION 's3://{S3_BUCKET}/predictions/'
    TBLPROPERTIES ('skip.header.line.count'='1');
    """
    _run_athena_query(ddl)
    logger.info(f"🏛️ Tabla Athena {ATHENA_TABLE} verificada/creada.")

def _run_athena_query(query: str):
    client = _get_session().client("athena")
    return client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_LOC}
    )

def update_actual_winner(year: int, round_num: int, winner_abbr: str):
    """Actualiza la columna actual_winner el lunes después de la carrera."""
    df = read_history_csv()
    mask = (df['year'] == year) & (df['round'] == round_num)
    
    if not df[mask].empty:
        df.loc[mask, 'actual_winner'] = (df.loc[mask, 'driver_abbr'] == winner_abbr).astype(int)
        append_to_history_csv(df[mask])
        logger.info(f"🏆 Ganador {winner_abbr} registrado para la ronda {round_num}.")
    else:
        logger.warning("⚠️ No se encontraron predicciones para actualizar el ganador.")