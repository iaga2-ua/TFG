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


import numpy as np

# ─── FastF1 setup ────────────────────────────────────────────────────────────
fastf1.Cache.enable_cache(FASTF1_CACHE)


# ─── Weather helpers ─────────────────────────────────────────────────────────

def _extract_weather_stats(session) -> dict:
    """
    Summarises weather data from a FastF1 session object into scalar features.
    The session must have been loaded with weather=True.

    Returns a dict with:
      rain_probability   – fraction of session timestamps with Rainfall == True
      is_wet_qualifying  – 1 if any rainfall was detected, else 0
      track_temp_c       – mean track temperature (°C)
      humidity_pct       – mean relative humidity (%)
      wind_speed_ms      – mean wind speed (m/s)
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

        # Weather features from qualifying session
        weather = _extract_weather_stats(session_quali)

        # Practice pace (FP2 long run + FP3 best lap)
        practice = fetch_practice_pace(year, round_num)
        # Build lookup: driver_abbr → {fp2_long_run_pace_gap_s, fp3_gap_to_best_s}
        practice_lookup = practice.set_index("driver_abbr").to_dict(orient="index")

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
                "points":          row.get("Points", 0),
                "status":          row.get("Status"),
                # Best quali time in seconds (try Q3 → Q2 → Q1)
                "best_quali_time": _safe_timedelta_s(
                    row.get("Q3Time") or row.get("Q2Time") or row.get("Q1Time")
                ),
                # Weather
                **weather,
                # Practice pace
                "fp2_long_run_pace_gap_s": p_data.get("fp2_long_run_pace_gap_s", np.nan),
                "fp3_gap_to_best_s":       p_data.get("fp3_gap_to_best_s",       np.nan),
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

    results["best_quali_time"] = results.apply(
        lambda r: _safe_timedelta_s(r.get("Q3") or r.get("Q2") or r.get("Q1")), axis=1
    )
    pole_time = results["best_quali_time"].min()
    results["quali_gap_to_pole_s"] = results["best_quali_time"] - pole_time

    results.rename(columns={
        "DriverNumber": "driver_number",
        "Abbreviation": "driver_abbr",
        "FullName":     "full_name",
        "TeamName":     "team",
        "GridPosition": "grid_position",
    }, inplace=True)

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

    keep_cols = [
        "year", "round", "event_name", "circuit",
        "driver_number", "driver_abbr", "full_name", "team",
        "grid_position", "quali_gap_to_pole_s",
        # weather
        "rain_probability", "is_wet_qualifying",
        "track_temp_c", "humidity_pct", "wind_speed_ms",
        # practice pace
        "fp2_long_run_pace_gap_s", "fp3_gap_to_best_s",
    ]
    return results[[c for c in keep_cols if c in results.columns]]


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
