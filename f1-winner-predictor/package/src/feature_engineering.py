"""
feature_engineering.py
======================
Transforma DataFrames crudos en matrices de características para XGBoost y TabNet.
Maneja historial de pilotos, standings del campeonato y metadata de circuitos.
"""

import logging
import numpy as np
import pandas as pd
from src.circuit_metadata import get_circuit_meta

logger = logging.getLogger(__name__)

# ─── HELPERS DE HISTORIAL (ROLLING FEATURES) ──────────────────────────────────

def _driver_rolling_avg_finish(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Media móvil de posición final (shift=1 para evitar fuga de datos)."""
    df = df.sort_values(["driver_abbr", "year", "round"])
    return df.groupby("driver_abbr")["race_position"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

def _driver_rolling_win_rate(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Tasa de victorias en las últimas N carreras."""
    df = df.sort_values(["driver_abbr", "year", "round"])
    return df.groupby("driver_abbr")["is_winner"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

def _driver_wet_win_rate(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Rendimiento específico en condiciones de lluvia."""
    df = df.sort_values(["driver_abbr", "year", "round"])
    
    def calc_wet_rate(g):
        wet_wins = (g["is_winner"] & g["is_wet_qualifying"].fillna(0).astype(bool)).astype(float)
        return wet_wins.shift(1).rolling(window, min_periods=1).mean()

    return df.groupby("driver_abbr").apply(calc_wet_rate).reset_index(level=0, drop=True)

# ─── METADATA Y STANDINGS ──────────────────────────────────────────────────────

def _attach_circuit_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Une la metadata del circuito al DataFrame.

    Si los datos proceden de data_collection (que ya llama a
    get_circuit_meta_from_fastf1), las columnas dinámicas
    (track_length_km, corner_count) y estáticas ya existen y no se
    sobreescriben.  Solo añade las columnas que falten, usando la tabla
    estática como fallback.
    """
    df = df.copy()
    # Columnas que esta función puede aportar
    meta_cols = [
        "overtake_difficulty", "drs_zones", "avg_safety_car_prob",
        "track_length_km", "corner_count",
    ]
    missing = [c for c in meta_cols if c not in df.columns]
    if not missing:
        return df  # todas las columnas ya vienen de data_collection

    meta_df = df["circuit"].apply(lambda x: pd.Series(get_circuit_meta(x)))
    # Añadir solo las que faltan (get_circuit_meta no devuelve track_length_km
    # ni corner_count, así que se rellenan con 0/NaN si no están presentes)
    for col in missing:
        if col in meta_df.columns:
            df[col] = meta_df[col].values
        elif col in ("track_length_km",):
            df[col] = float("nan")
        else:
            df[col] = 0
    return df

def _circuit_driver_history(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas históricas del piloto en este circuito específico."""
    df = df.sort_values(["year", "round"])
    results = []
    
    for (year, rnd), grp in df.groupby(["year", "round"]):
        hist = df[(df["year"] < year) | ((df["year"] == year) & (df["round"] < rnd))]
        
        if hist.empty:
            for abbr in grp["driver_abbr"]:
                results.append({"year": year, "round": rnd, "driver_abbr": abbr,
                               "driver_best_finish_circuit": np.nan, "driver_avg_finish_circuit": np.nan})
            continue

        circuit_val = grp["circuit"].iloc[0]
        circ_hist = hist[hist["circuit"] == circuit_val]
        stats = circ_hist.groupby("driver_abbr")["race_position"].agg(
            driver_best_finish_circuit="min", driver_avg_finish_circuit="mean"
        ).reset_index()

        merged = grp[["driver_abbr"]].merge(stats, on="driver_abbr", how="left")
        merged["year"], merged["round"] = year, rnd
        results.append(merged)

    return pd.concat(results, ignore_index=True)

def _championship_standings(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula puntos y posición en el mundial ANTES de la carrera actual."""
    df = df.sort_values(["year", "round"])
    rows_d, rows_c = [], []

    for (year, rnd), grp in df.groupby(["year", "round"]):
        hist = df[(df["year"] == year) & (df["round"] < rnd)]
        
        # Si es la Ronda 1, todos empiezan de 0
        if hist.empty:
            for _, r in grp.iterrows():
                rows_d.append({"year": year, "round": rnd, "driver_abbr": r["driver_abbr"], 
                               "driver_champ_pos": 10, "driver_champ_points": 0.0})
                rows_c.append({"year": year, "round": rnd, "team": r["team"], 
                               "constructor_champ_pos": 5, "constructor_champ_points": 0.0})
            continue

        # Standings de Pilotos
        d_pts = hist.groupby("driver_abbr")["points"].sum().sort_values(ascending=False).reset_index()
        d_pts["driver_champ_pos"] = d_pts.index + 1
        
        # Standings de Constructores
        c_pts = hist.groupby("team")["points"].sum().sort_values(ascending=False).reset_index()
        c_pts["constructor_champ_pos"] = c_pts.index + 1

        for _, r in grp.iterrows():
            d_match = d_pts[d_pts["driver_abbr"] == r["driver_abbr"]]
            rows_d.append({
                "year": year, "round": rnd, "driver_abbr": r["driver_abbr"],
                "driver_champ_pos": d_match["driver_champ_pos"].values[0] if not d_match.empty else 20,
                "driver_champ_points": d_match["points"].values[0] if not d_match.empty else 0.0
            })
            c_match = c_pts[c_pts["team"] == r["team"]]
            rows_c.append({
                "year": year, "round": rnd, "team": r["team"],
                "constructor_champ_pos": c_match["constructor_champ_pos"].values[0] if not c_match.empty else 11,
                "constructor_champ_points": c_match["points"].values[0] if not c_match.empty else 0.0
            })

    return pd.DataFrame(rows_d), pd.DataFrame(rows_c)

# ─── LABEL ENCODING ───────────────────────────────────────────────────────────

def apply_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    df["circuit_encoded"] = df["circuit"].map(encoders.get("circuit", {})).fillna(-1).astype(int)
    df["constructor_encoded"] = df["team"].map(encoders.get("team", {})).fillna(-1).astype(int)
    return df

# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────



def build_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Pipeline completo para ENTRENAMIENTO."""
    df = df_raw.copy()
    
    # Conversión de tipos y limpieza
    for col in ["race_position", "grid_position", "points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Añadir features de historial
    df["driver_avg_finish_l3"] = _driver_rolling_avg_finish(df)
    df["driver_win_rate_l5"] = _driver_rolling_win_rate(df)
    df["driver_wet_win_rate"] = _driver_wet_win_rate(df)
    
    # Añadir circuitos y standings
    circ_hist = _circuit_driver_history(df)
    df = df.merge(circ_hist, on=["year", "round", "driver_abbr"], how="left")
    df = _attach_circuit_metadata(df)
    
    d_champ, c_champ = _championship_standings(df)
    df = df.merge(d_champ, on=["year", "round", "driver_abbr"], how="left")
    df = df.merge(c_champ, on=["year", "round", "team"], how="left")

    # Encoders
    encoders = {
        "circuit": {v: i for i, v in enumerate(sorted(df["circuit"].unique()))},
        "team": {v: i for i, v in enumerate(sorted(df["team"].unique()))}
    }
    df = apply_label_encoders(df, encoders)

    from config import FEATURE_COLS, TARGET_COL
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]
    
    return X, y, encoders

def apply_features(df_live: pd.DataFrame, encoders: dict, history_df: pd.DataFrame = None) -> pd.DataFrame:
    """Pipeline para INFERENCIA (Sábado de Quali)."""
    from config import FEATURE_COLS
    df = df_live.copy()
    
    # Inyectar metadata estática
    df = _attach_circuit_metadata(df)
    
    if history_df is not None:
        # Aquí calculamos los valores finales del historial basándonos en history_df
        # para que el modelo sepa cómo viene cada piloto a esta carrera.
        # (Lógica simplificada para brevedad, similar a build_features)
        df = _enrich_live_from_history(df, history_df)

    df = apply_label_encoders(df, encoders)
    
    # Asegurar que todas las columnas existen
    for col in FEATURE_COLS:
        if col not in df.columns: df[col] = 0
        
    return df[FEATURE_COLS].fillna(0)

def _enrich_live_from_history(df, hist):
    """Auxiliar para rellenar datos de historial en tiempo real."""
    # Ejemplo para driver_avg_finish_l3
    for abbr in df["driver_abbr"].unique():
        driver_hist = hist[hist["driver_abbr"] == abbr].sort_values(["year", "round"])
        if not driver_hist.empty:
            df.loc[df["driver_abbr"] == abbr, "driver_avg_finish_l3"] = driver_hist["race_position"].tail(3).mean()
            df.loc[df["driver_abbr"] == abbr, "driver_win_rate_l5"] = driver_hist["is_winner"].tail(5).mean()
    return df