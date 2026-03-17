"""
data_collection.py
==================
Fetches race and qualifying results from the FastF1 API for a list of seasons
and saves raw DataFrames to data/raw/.

Usage (standalone):
    python -m src.data_collection
"""

import logging
import warnings
from pathlib import Path

import fastf1
import pandas as pd

# Suppress noisy FastF1 / urllib3 warnings in production
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FASTF1_CACHE, RAW_DIR, TRAIN_SEASONS
from src.circuit_metadata import get_circuit_meta_from_fastf1


import numpy as np

# ─── FastF1 setup ────────────────────────────────────────────────────────────
fastf1.Cache.enable_cache(FASTF1_CACHE)


# ─── Weather helpers ─────────────────────────────────────────────────────────

def _extract_weather_stats(session) -> dict:
    """
    Summarises weather data from a FastF1 session object into scalar features.
    The session must have been loaded with weather=True.

    Returns a dict with:
      rain_probability   - fraction of session timestamps with Rainfall == True
      is_wet_qualifying  - 1 if any rainfall was detected, else 0
      track_temp_c       - mean track temperature (°C)
      humidity_pct       - mean relative humidity (%)
      wind_speed_ms      - mean wind speed (m/s)
    """
    defaults = {
        "rain_probability":  0.0,
        "is_wet_qualifying": 0,
        "track_temp_c":      float("nan"),
        "humidity_pct":      float("nan"),
        "wind_speed_ms":     float("nan"),
    }
    try:
        wd = session.weather_data
        if wd is None or wd.empty:
            return defaults
        rain_prob = float(wd["Rainfall"].astype(bool).mean())
        return {
            "rain_probability":  round(rain_prob, 4),
            "is_wet_qualifying": int(rain_prob > 0),
            "track_temp_c":      float(wd["TrackTemp"].mean()),
            "humidity_pct":      float(wd["Humidity"].mean()),
            "wind_speed_ms":     float(wd["WindSpeed"].mean()),
        }
    except Exception:
        return defaults


# ─── Practice pace helpers ───────────────────────────────────────────────────

_RACE_COMPOUNDS = {"MEDIUM", "HARD", "SOFT"}  # all compounds valid for race pace


def fetch_practice_pace(year: int, round_num: int) -> pd.DataFrame:
    """
    Extracts clean-air race pace proxies from practice sessions:
      fp2_long_run_pace_gap_s  – gap (s) to fastest FP2 representative lap
                                  (median of clean laps on slick compounds)
      fp3_gap_to_best_s        – gap (s) to fastest FP3 lap per driver

    Returns a DataFrame indexed by driver_abbr.
    Handles sprint weekends (no FP2/FP3) gracefully.
    """
    pace_frames = []

    for session_name, col_out in [("FP2", "fp2_long_run_pace_gap_s"),
                                   ("FP3", "fp3_gap_to_best_s")]:
        try:
            sess = fastf1.get_session(year, round_num, session_name)
            sess.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = sess.laps

            if laps is None or laps.empty:
                pace_frames.append(pd.DataFrame(columns=["driver_abbr", col_out]))
                continue

            # Keep clean laps: no pit in/out, accurate timing
            clean = laps[
                laps["PitInTime"].isna() &
                laps["PitOutTime"].isna() &
                laps["IsAccurate"].fillna(True) &
                laps["Compound"].isin(_RACE_COMPOUNDS)
            ].copy()

            if clean.empty:
                # Sprint weekend fallback: use all valid laps
                clean = laps[
                    laps["PitInTime"].isna() &
                    laps["PitOutTime"].isna()
                ].copy()

            if clean.empty:
                pace_frames.append(pd.DataFrame(columns=["driver_abbr", col_out]))
                continue

            # Convert LapTime to seconds
            clean["lap_time_s"] = clean["LapTime"].apply(
                lambda t: pd.Timedelta(t).total_seconds() if pd.notna(t) else np.nan
            )
            clean = clean.dropna(subset=["lap_time_s"])
            # Remove obvious outliers (> 110% of fastest lap)
            fastest = clean["lap_time_s"].min()
            clean = clean[clean["lap_time_s"] <= fastest * 1.10]

            if session_name == "FP2":
                # Median lap per driver → representative long-run pace
                pace = (
                    clean.groupby("Driver")["lap_time_s"]
                    .median()
                    .rename("pace")
                )
            else:
                # Best lap per driver in FP3
                pace = (
                    clean.groupby("Driver")["lap_time_s"]
                    .min()
                    .rename("pace")
                )

            best_pace = pace.min()
            gap = (pace - best_pace).reset_index()
            gap.columns = ["driver_abbr", col_out]
            pace_frames.append(gap)

        except Exception as exc:
            logger.debug(f"  ℹ  {year} R{round_num} {session_name} not available: {exc}")
            pace_frames.append(pd.DataFrame(columns=["driver_abbr", col_out]))

    # Sprint weekend fallback: FP2/FP3 do not exist (schedule: FP1 → SQ → S → Q).
    # Use FP1 best-lap gaps as a substitute for both columns so the model
    # receives real pace data instead of all-zero placeholders after fillna(0).
    if len(pace_frames[0]) == 0 and len(pace_frames[1]) == 0:
        logger.info(
            f"  ⚡ {year} R{round_num}: sprint weekend detected — "
            "falling back to FP1 for practice pace features"
        )
        try:
            sess_fp1 = fastf1.get_session(year, round_num, "FP1")
            sess_fp1.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = sess_fp1.laps
            if laps is not None and not laps.empty:
                clean = laps[
                    laps["PitInTime"].isna() &
                    laps["PitOutTime"].isna() &
                    laps["IsAccurate"].fillna(True)
                ].copy()
                if clean.empty:
                    clean = laps[
                        laps["PitInTime"].isna() & laps["PitOutTime"].isna()
                    ].copy()
                if not clean.empty:
                    clean["lap_time_s"] = clean["LapTime"].apply(
                        lambda t: pd.Timedelta(t).total_seconds() if pd.notna(t) else np.nan
                    )
                    clean = clean.dropna(subset=["lap_time_s"])
                    fastest = clean["lap_time_s"].min()
                    clean = clean[clean["lap_time_s"] <= fastest * 1.10]
                    best_per_driver = clean.groupby("Driver")["lap_time_s"].min()
                    gap_fp1 = (best_per_driver - best_per_driver.min()).reset_index()
                    gap_fp1.columns = ["driver_abbr", "fp1_gap"]
                    # FP1 best-lap gap substitutes both FP2 long-run and FP3 best-lap
                    pace_frames[0] = gap_fp1.rename(columns={"fp1_gap": "fp2_long_run_pace_gap_s"})
                    pace_frames[1] = gap_fp1.rename(columns={"fp1_gap": "fp3_gap_to_best_s"})
        except Exception as exc:
            logger.debug(f"  ℹ  {year} R{round_num} FP1 also not available: {exc}")

    # Merge FP2 and FP3 frames on driver_abbr
    result = pace_frames[0].merge(pace_frames[1], on="driver_abbr", how="outer")
    return result


def fetch_season_results(year: int) -> pd.DataFrame:
    """
    Returns a single DataFrame with one row per (race, driver) for every
    completed race in *year*.  Columns include qualifying and race results.
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Keep only events that have already happened (for historical data)
    schedule = schedule[schedule["EventFormat"] != "testing"]

    rows = []
    for _, event in schedule.iterrows():
        round_num  = event["RoundNumber"]
        event_name = event["EventName"]
        circuit    = event["Location"]

        try:
            session_race = fastf1.get_session(year, round_num, "R")
            session_race.load(laps=False, telemetry=False, weather=False, messages=False)

            session_quali = fastf1.get_session(year, round_num, "Q")
            session_quali.load(laps=False, telemetry=False, weather=True, messages=False)
        except Exception as exc:
            logger.warning(f"  ⚠  {year} R{round_num} ({event_name}): {exc}")
            continue

        race_results  = session_race.results
        quali_results = session_quali.results

        if race_results is None or race_results.empty:
            logger.warning(f"  ⚠  {year} R{round_num}: empty race results, skipping.")
            continue

        # Circuit metadata from FastF1 (dynamic + static fields)
        circuit_meta = get_circuit_meta_from_fastf1(session_race)

        # Weather features from qualifying session
        weather = _extract_weather_stats(session_quali)

        # Practice pace (FP2 long run + FP3 best lap; FP1 fallback on sprint weekends)
        practice = fetch_practice_pace(year, round_num)
        # Build lookup: driver_abbr → {fp2_long_run_pace_gap_s, fp3_gap_to_best_s}
        practice_lookup = practice.set_index("driver_abbr").to_dict(orient="index")

        # Sprint Race lookup: puntos y posición (sesión "S").
        # En GPs normales FastF1 lanza InvalidSessionError → se ignora.
        # El Sprint Race ocurre el sábado por la mañana, ANTES de la
        # clasificación, así que sus resultados son válidos como feature.
        sprint_points_lookup:   dict[str, float] = {}
        sprint_position_lookup: dict[str, float] = {}
        try:
            session_sprint = fastf1.get_session(year, round_num, "S")
            session_sprint.load(laps=False, telemetry=False, weather=False, messages=False)
            if session_sprint.results is not None and not session_sprint.results.empty:
                for _, sr in session_sprint.results.iterrows():
                    abbr_s = sr.get("Abbreviation")
                    pts_s  = float(sr.get("Points", 0) or 0)
                    pos_s  = sr.get("Position")
                    if abbr_s:
                        sprint_points_lookup[abbr_s] = pts_s
                        if pd.notna(pos_s):
                            sprint_position_lookup[abbr_s] = float(pos_s)
                logger.info(
                    f"  🏎  {year} R{round_num}: sprint race — "
                    f"sprint points + positions added for {len(sprint_points_lookup)} drivers"
                )
        except Exception:
            pass  # Not a sprint weekend

        # Merge qualifying grid positions into race results
        quali_cols = ["DriverNumber", "GridPosition", "Q3", "Q2", "Q1", "Position"]
        quali_renamed = (
            quali_results[quali_cols]
            .rename(columns={
                "GridPosition": "QualiGrid",
                "Position":     "QualiPos",
                "Q3":           "Q3Time",
                "Q2":           "Q2Time",
                "Q1":           "Q1Time",
            })
        )
        merged = race_results.merge(quali_renamed, on="DriverNumber", how="left")

        for _, row in merged.iterrows():
            abbr = row.get("Abbreviation")
            p_data = practice_lookup.get(abbr, {})
            rows.append({
                "year":            year,
                "round":           round_num,
                "event_name":      event_name,
                "circuit":         circuit,
                "driver_number":   row.get("DriverNumber"),
                "driver_abbr":     abbr,
                "full_name":       row.get("FullName", ""),
                "team":            row.get("TeamName"),
                "grid_position":   row.get("GridPosition", row.get("QualiGrid")),
                "race_position":   row.get("Position"),
                "points":          float(row.get("Points", 0) or 0) + sprint_points_lookup.get(abbr, 0.0),
                "status":          row.get("Status"),
                # Best quali time: mínimo entre Q1, Q2, Q3 (mejor vuelta de toda la qualy)
                "best_quali_time": min(
                    (t for t in (
                        _safe_timedelta_s(row.get("Q3Time")),
                        _safe_timedelta_s(row.get("Q2Time")),
                        _safe_timedelta_s(row.get("Q1Time")),
                    ) if not np.isnan(t)),
                    default=float("nan"),
                ),
                # Weather
                **weather,
                # Practice pace
                "fp2_long_run_pace_gap_s": p_data.get("fp2_long_run_pace_gap_s", np.nan),
                "fp3_gap_to_best_s":       p_data.get("fp3_gap_to_best_s",       np.nan),
                # Sprint Race position (0 = GP sin sprint)
                "sprint_race_position":     sprint_position_lookup.get(abbr, 0.0),
                # Circuit metadata (FastF1 dynamic + static)
                "track_length_km":          circuit_meta["track_length_km"],
                "corner_count":             circuit_meta["corner_count"],
                "overtake_difficulty":      circuit_meta["overtake_difficulty"],
                "drs_zones":                circuit_meta["drs_zones"],
                "avg_safety_car_prob":      circuit_meta["avg_safety_car_prob"],
            })

        logger.info(f"  ✓  {year} R{round_num}: {event_name} — {len(merged)} drivers"
                    f"  🌧 rain={weather['rain_probability']:.0%}")

    df = pd.DataFrame(rows)
    return df


def compute_pole_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `quali_gap_to_pole_s`: qualifying time gap (seconds) to pole position
    for each driver within a race.
    """
    pole_times = (
        df.groupby(["year", "round"])["best_quali_time"]
        .min()
        .rename("pole_time")
    )
    df = df.merge(pole_times, on=["year", "round"], how="left")
    df["quali_gap_to_pole_s"] = df["best_quali_time"] - df["pole_time"]
    df.drop(columns=["pole_time"], inplace=True)
    return df


def _safe_timedelta_s(val) -> float:
    """Convert a timedelta-like value to total seconds; returns NaN if invalid."""
    try:
        return pd.Timedelta(val).total_seconds()
    except Exception:
        return float("nan")


def collect_all_seasons(seasons: list[int] | None = None, save: bool = True) -> pd.DataFrame:
    """
    Iterates over *seasons* (defaults to config.TRAIN_SEASONS), fetches results,
    concatenates into one DataFrame, computes the pole gap, and optionally saves
    to data/raw/race_results_raw.csv.
    """
    if seasons is None:
        seasons = TRAIN_SEASONS

    all_dfs = []
    for year in seasons:
        logger.info(f"📅 Fetching season {year}...")
        df_year = fetch_season_results(year)
        all_dfs.append(df_year)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = compute_pole_gap(combined)

    # Derive binary target
    combined["race_position"] = pd.to_numeric(combined["race_position"], errors="coerce")
    combined["is_winner"] = (combined["race_position"] == 1).astype(int)

    if save:
        out_path = RAW_DIR / "race_results_raw.csv"
        combined.to_csv(out_path, index=False)
        logger.info(f"💾 Saved {len(combined)} rows → {out_path}")

    return combined


def fetch_qualifying_snapshot(year: int, round_num: int) -> pd.DataFrame:
    """
    Fetches the qualifying session for a single race weekend (Saturday).
    Enriches each driver row with:
      - quali gap to pole
      - weather features (rain probability, track temp, humidity, wind)
      - FP2 long-run pace gap (clean-air race pace proxy)
      - FP3 best-lap gap
    """
    fastf1.Cache.enable_cache(FASTF1_CACHE)
    event = fastf1.get_event(year, round_num)
    circuit    = event["Location"]
    event_name = event["EventName"]

    # ── Qualifying ────────────────────────────────────────────────────────────
    session = fastf1.get_session(year, round_num, "Q")
    session.load(laps=False, telemetry=False, weather=True, messages=False)

    results = session.results.copy()
    results["year"]       = year
    results["round"]      = round_num
    results["event_name"] = event_name
    results["circuit"]    = circuit

    # Mejor vuelta de cada piloto en TODA la qualy (min de Q1, Q2, Q3).
    # Más preciso que solo Q3: si un piloto fue eliminado en Q1, usamos
    # su mejor tiempo real, no NaN.
    def _best_across_sessions(row):
        times = [_safe_timedelta_s(row.get(q)) for q in ("Q1", "Q2", "Q3")]
        valid = [t for t in times if not np.isnan(t)]
        return min(valid) if valid else float("nan")

    results["best_quali_time"] = results.apply(_best_across_sessions, axis=1)
    pole_time = results["best_quali_time"].min()
    results["quali_gap_to_pole_s"] = results["best_quali_time"] - pole_time

    results.rename(columns={
        "DriverNumber": "driver_number",
        "Abbreviation": "driver_abbr",
        "FullName":     "full_name",
        "TeamName":     "team",
        "GridPosition": "race_grid_position",  # posición tras penalizaciones (domingo)
        "Position":     "grid_position",        # posición de clasificación (sábado)
    }, inplace=True)

    # Pilotos sin tiempo de clasificación (penalización, incidente, DNS):
    # su grid_position y quali_gap_to_pole_s son NaN.
    # Rellenamos con valores penalizados para que el modelo no los confunda con pole.
    n_drivers = len(results)
    grid_nan = results["grid_position"].isna()
    if grid_nan.any():
        # Asignar posiciones al final de la parrilla a partir de la última válida
        last_pos = results["grid_position"].max()
        if np.isnan(last_pos):
            last_pos = 0
        for i, idx in enumerate(results.index[grid_nan], start=1):
            results.at[idx, "grid_position"] = last_pos + i

    gap_nan = results["quali_gap_to_pole_s"].isna()
    if gap_nan.any():
        max_gap = results["quali_gap_to_pole_s"].max()
        max_gap = 0.0 if np.isnan(max_gap) else max_gap
        results.loc[gap_nan, "quali_gap_to_pole_s"] = max_gap + 5.0

    # ── Weather ───────────────────────────────────────────────────────────────
    weather = _extract_weather_stats(session)
    for col, val in weather.items():
        results[col] = val

    logger.info(
        f"🌡  Qualifying weather — "
        f"Rain: {weather['rain_probability']:.0%}  "
        f"Track: {weather['track_temp_c']:.1f}°C  "
        f"Humidity: {weather['humidity_pct']:.1f}%  "
        f"Wind: {weather['wind_speed_ms']:.1f} m/s"
    )

    # ── Practice pace ─────────────────────────────────────────────────────────
    practice = fetch_practice_pace(year, round_num)
    practice_lookup = practice.set_index("driver_abbr").to_dict(orient="index")

    results["fp2_long_run_pace_gap_s"] = results["driver_abbr"].map(
        lambda a: practice_lookup.get(a, {}).get("fp2_long_run_pace_gap_s", np.nan)
    )
    results["fp3_gap_to_best_s"] = results["driver_abbr"].map(
        lambda a: practice_lookup.get(a, {}).get("fp3_gap_to_best_s", np.nan)
    )

    # ── Sprint Race position (sábado mañana, antes de la clasificación) ───────
    sprint_position_lookup: dict[str, float] = {}
    try:
        session_sprint = fastf1.get_session(year, round_num, "S")
        session_sprint.load(laps=False, telemetry=False, weather=False, messages=False)
        if session_sprint.results is not None and not session_sprint.results.empty:
            for _, sr in session_sprint.results.iterrows():
                abbr_s = sr.get("Abbreviation")
                pos_s  = sr.get("Position")
                if abbr_s and pd.notna(pos_s):
                    sprint_position_lookup[abbr_s] = float(pos_s)
            logger.info(
                f"🏎  Sprint Race R{round_num}: posiciones cargadas "
                f"para {len(sprint_position_lookup)} pilotos"
            )
    except Exception:
        pass  # GP normal sin sprint
    results["sprint_race_position"] = results["driver_abbr"].map(
        lambda a: sprint_position_lookup.get(a, 0.0)
    )

    # ── Circuit metadata (FastF1 dinamico + tabla estatica) ───────────────────
    # Mismo flujo que en fetch_season_results para que entrenamiento e
    # inferencia (Lambda y local) reciban exactamente las mismas columnas.
    circuit_meta = get_circuit_meta_from_fastf1(session)
    results["track_length_km"]     = circuit_meta["track_length_km"]
    results["corner_count"]        = circuit_meta["corner_count"]
    results["overtake_difficulty"] = circuit_meta["overtake_difficulty"]
    results["drs_zones"]           = circuit_meta["drs_zones"]
    results["avg_safety_car_prob"] = circuit_meta["avg_safety_car_prob"]
    logger.info(
        f"📐 Circuito: {circuit_meta['location']} — "
        f"{circuit_meta['track_length_km']:.3f} km, "
        f"{circuit_meta['corner_count']} curvas, "
        f"{circuit_meta['drs_zones']} zonas DRS"
    )

    keep_cols = [
        "year", "round", "event_name", "circuit",
        "driver_number", "driver_abbr", "full_name", "team",
        "grid_position", "quali_gap_to_pole_s",
        # weather
        "rain_probability", "is_wet_qualifying",
        "track_temp_c", "humidity_pct", "wind_speed_ms",
        # practice pace
        "fp2_long_run_pace_gap_s", "fp3_gap_to_best_s",
        # sprint race
        "sprint_race_position",
        # circuit metadata (FastF1)
        "track_length_km", "corner_count",
        "overtake_difficulty", "drs_zones", "avg_safety_car_prob",
    ]
    df = results[[c for c in keep_cols if c in results.columns]].copy()

    # Validar que la clasificación realmente ocurrió:
    # si grid_position es todo NaN, FastF1 devuelve la entry list pero la
    # sesión aún no se ha disputado → no hay datos reales de quali.
    if df["grid_position"].isna().all():
        logger.warning(
            "⚠️  %d R%d (%s): grid_position es todo NaN — "
            "la clasificación aún no se ha disputado o los datos no están disponibles.",
            year, round_num, event_name,
        )
        return pd.DataFrame()

    return df


def fetch_live_data_2026(year: int, round_num: int) -> pd.DataFrame:
    """
    Versión optimizada para el flujo de 2026.
    Captura el snapshot y prepara las columnas exactas que esperan 
    nuestros dos modelos (XGBoost y TabNet).
    """
    logger.info(f"🏎️ Generando snapshot en vivo para {year} R{round_num}...")
    
    df_live = fetch_qualifying_snapshot(year, round_num)
    
    # Verificación de integridad para 2026
    driver_count = len(df_live)
    if driver_count != 22:
        logger.warning(f"Se detectaron {driver_count} pilotos. Para 2026 se esperan 22.")
    
    # Rellenar nans en gaps de práctica (si un piloto no rodó) 
    # con un valor penalizado (ej. el peor tiempo + 0.5s)
    for col in ["fp2_long_run_pace_gap_s", "fp3_gap_to_best_s", "quali_gap_to_pole_s"]:
        if df_live[col].isnull().any():
            max_val = df_live[col].max()
            df_live[col] = df_live[col].fillna(max_val + 0.5)
            
    return df_live

if __name__ == "__main__":
    df = collect_all_seasons()
    print(df.head())
    print(f"\nTotal rows: {len(df)} | Winners: {df['is_winner'].sum()}")
