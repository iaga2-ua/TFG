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

def _driver_current_season_win_rate(df: pd.DataFrame) -> pd.Series:
    """Tasa de victorias del piloto en la temporada actual, usando solo carreras anteriores."""
    df = df.sort_values(["driver_abbr", "year", "round"])

    def calc(g):
        result = pd.Series(0.0, index=g.index)
        for i, (idx, row) in enumerate(g.iterrows()):
            season_past = g[(g["year"] == row["year"]) & (g["round"] < row["round"])]
            if season_past.empty:
                result.at[idx] = 0.0
            else:
                result.at[idx] = season_past["is_winner"].mean()
        return result

    return df.groupby("driver_abbr").apply(calc).reset_index(level=0, drop=True)

def _driver_races_since_last_win(df: pd.DataFrame) -> pd.Series:
    """Número de carreras desde la última victoria (50 si nunca ha ganado o lleva mucho)."""
    df = df.sort_values(["driver_abbr", "year", "round"])

    def calc(g):
        result = pd.Series(50, index=g.index, dtype=float)
        for i, (idx, row) in enumerate(g.iterrows()):
            past_wins = g[(g["year"] < row["year"]) |
                          ((g["year"] == row["year"]) & (g["round"] < row["round"]))]
            past_wins = past_wins[past_wins["is_winner"] == 1]
            if past_wins.empty:
                result.at[idx] = 50.0
            else:
                # Número de carreras disputadas desde la última victoria
                last_win_idx = past_wins.index[-1]
                races_after = g[(g["year"] > g.at[last_win_idx, "year"]) |
                                 ((g["year"] == g.at[last_win_idx, "year"]) &
                                  (g["round"] > g.at[last_win_idx, "round"]))]
                races_before_now = races_after[
                    (races_after["year"] < row["year"]) |
                    ((races_after["year"] == row["year"]) & (races_after["round"] < row["round"]))
                ]
                result.at[idx] = float(len(races_before_now))
        return result

    return df.groupby("driver_abbr").apply(calc).reset_index(level=0, drop=True)

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
            empty_df = grp[["driver_abbr"]].copy()
            empty_df["driver_best_finish_circuit"] = np.nan
            empty_df["driver_avg_finish_circuit"]  = np.nan
            empty_df["year"]  = year
            empty_df["round"] = rnd
            results.append(empty_df)
            continue

        circuit_val = grp["circuit"].iloc[0]
        # Limitar a los últimos 3 años: contexto reciente más relevante que dominancia histórica
        circ_hist = hist[(hist["circuit"] == circuit_val) & (hist["year"] >= year - 3)]
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
    df["driver_current_season_win_rate"] = _driver_current_season_win_rate(df)
    df["driver_races_since_last_win"] = _driver_races_since_last_win(df)
    
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

    # race_number es simplemente el número de ronda en la temporada
    df["race_number"] = df["round"]

    from config import FEATURE_COLS, TARGET_COL
    X = df[FEATURE_COLS].fillna(0).reset_index(drop=True)
    y = df[TARGET_COL].reset_index(drop=True)
    ctx = df[["driver_abbr", "year", "round"]].reset_index(drop=True)

    return X, y, encoders, ctx

def apply_features(
    df_live: pd.DataFrame,
    encoders: dict,
    history_df: pd.DataFrame = None,
    year: int = None,
    round_num: int = None,
) -> pd.DataFrame:
    """Pipeline para INFERENCIA (Sábado de Quali).

    history_df debe ser race_results_raw.csv (descargado de S3).
    year / round_num delimitan qué datos históricos son válidos
    (sólo carreras ANTERIORES a la que se está prediciendo).
    """
    from config import FEATURE_COLS
    df = df_live.copy()

    # Inyectar metadata estática
    df = _attach_circuit_metadata(df)

    if history_df is not None and not history_df.empty and "driver_abbr" in history_df.columns:
        df = _enrich_live_from_history(df, history_df, year=year, round_num=round_num)

    df = apply_label_encoders(df, encoders)

    # Asegurar que todas las columnas existen
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_COLS].fillna(0)


def _enrich_live_from_history(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    year: int = None,
    round_num: int = None,
) -> pd.DataFrame:
    """Calcula features históricas usando race_results_raw para cada piloto.

    Solo usa carreras anteriores a (year, round_num) para evitar data leakage.
    Si year/round_num son None, usa todo el historial.
    """
    df = df.copy()
    hist = hist.copy()
    hist["race_position"] = pd.to_numeric(hist["race_position"], errors="coerce")
    if "is_winner" not in hist.columns:
        hist["is_winner"] = (hist["race_position"] == 1).astype(int)

    # Filtrar: solo datos anteriores a la carrera que se predice
    if year is not None and round_num is not None:
        past = hist[
            (hist["year"] < year) |
            ((hist["year"] == year) & (hist["round"] < round_num))
        ].sort_values(["year", "round"])
    else:
        past = hist.sort_values(["year", "round"])

    circuit_val = df["circuit"].iloc[0] if "circuit" in df.columns else None

    for abbr in df["driver_abbr"].unique():
        drv = past[past["driver_abbr"] == abbr]

        # Rolling features
        df.loc[df["driver_abbr"] == abbr, "driver_avg_finish_l3"] = (
            drv["race_position"].tail(3).mean() if not drv.empty else np.nan
        )
        df.loc[df["driver_abbr"] == abbr, "driver_win_rate_l5"] = (
            drv["is_winner"].tail(5).mean() if not drv.empty else 0.0
        )
        wet_drv = drv[drv.get("is_wet_qualifying", pd.Series(0, index=drv.index)).fillna(0).astype(bool)]
        df.loc[df["driver_abbr"] == abbr, "driver_wet_win_rate"] = (
            wet_drv["is_winner"].tail(10).mean() if not wet_drv.empty else 0.0
        )

        # Circuit-specific history (últimos 3 años: excluye dominancia obsoleta)
        if circuit_val is not None:
            year_cutoff = (year if year is not None else int(past["year"].max())) - 3
            circ_drv = drv[(drv["circuit"] == circuit_val) & (drv["year"] >= year_cutoff)]
            df.loc[df["driver_abbr"] == abbr, "driver_best_finish_circuit"] = (
                circ_drv["race_position"].min() if not circ_drv.empty else np.nan
            )
            df.loc[df["driver_abbr"] == abbr, "driver_avg_finish_circuit"] = (
                circ_drv["race_position"].mean() if not circ_drv.empty else np.nan
            )

        # Current-season form (solo carreras de la temporada que se predice)
        if year is not None:
            season_drv = drv[drv["year"] == year]
        else:
            season_drv = drv
        df.loc[df["driver_abbr"] == abbr, "driver_current_season_win_rate"] = (
            season_drv["is_winner"].mean() if not season_drv.empty else 0.0
        )
        # Races since last win (overall, not just this season)
        wins = drv[drv["is_winner"] == 1]
        if wins.empty:
            df.loc[df["driver_abbr"] == abbr, "driver_races_since_last_win"] = 50.0
        else:
            last_win_year  = wins.iloc[-1]["year"]
            last_win_round = wins.iloc[-1]["round"]
            races_after = len(drv[
                (drv["year"] > last_win_year) |
                ((drv["year"] == last_win_year) & (drv["round"] > last_win_round))
            ])
            df.loc[df["driver_abbr"] == abbr, "driver_races_since_last_win"] = float(races_after)

    # Championship standings (sólo carreras del año actual hasta la ronda anterior)
    if year is not None and round_num is not None:
        season_past = past[past["year"] == year]
    else:
        season_past = past

    if not season_past.empty:
        d_pts = (
            season_past.groupby("driver_abbr")["points"].sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        d_pts["driver_champ_pos"] = range(1, len(d_pts) + 1)

        teams = season_past[["driver_abbr", "team"]].drop_duplicates("driver_abbr")
        c_pts = (
            season_past.groupby("team")["points"].sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        c_pts["constructor_champ_pos"] = range(1, len(c_pts) + 1)

        for abbr in df["driver_abbr"].unique():
            row_d = d_pts[d_pts["driver_abbr"] == abbr]
            df.loc[df["driver_abbr"] == abbr, "driver_champ_points"] = (
                row_d["points"].values[0] if not row_d.empty else 0.0
            )
            df.loc[df["driver_abbr"] == abbr, "driver_champ_pos"] = (
                row_d["driver_champ_pos"].values[0] if not row_d.empty else 20
            )
            team_row = teams[teams["driver_abbr"] == abbr]
            if not team_row.empty:
                team_name = team_row["team"].values[0]
                row_c = c_pts[c_pts["team"] == team_name]
                df.loc[df["driver_abbr"] == abbr, "constructor_champ_points"] = (
                    row_c["points"].values[0] if not row_c.empty else 0.0
                )
                df.loc[df["driver_abbr"] == abbr, "constructor_champ_pos"] = (
                    row_c["constructor_champ_pos"].values[0] if not row_c.empty else 11
                )

    return df