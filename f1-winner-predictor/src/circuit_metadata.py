"""
circuit_metadata.py
===================
Static characteristics for each F1 circuit.
These are features the FastF1 API cannot provide dynamically.

overtake_difficulty : int (1–5)
    1 = very easy to overtake (Monza, Bahrain)
    5 = nearly impossible (Monaco, Baku street section)

drs_zones : int
    Number of DRS activation zones at the circuit.

avg_safety_car_prob : float (0–1)
    Historical probability of a safety car / VSC during the race.
    Approximate values based on 2018-2025 data.
"""

# key = FastF1 circuit Location string (as returned by event["Location"])
CIRCUIT_META: dict[str, dict] = {
    # ── Bahrain ───────────────────────────────────────────────────────────────
    "Sakhir": {
        "overtake_difficulty": 1,
        "drs_zones": 3,
        "avg_safety_car_prob": 0.25,
    },
    # ── Saudi Arabia ──────────────────────────────────────────────────────────
    "Jeddah": {
        "overtake_difficulty": 2,
        "drs_zones": 3,
        "avg_safety_car_prob": 0.55,
    },
    # ── Australia ─────────────────────────────────────────────────────────────
    "Melbourne": {
        "overtake_difficulty": 2,
        "drs_zones": 4,
        "avg_safety_car_prob": 0.60,
    },
    # ── Japan ─────────────────────────────────────────────────────────────────
    "Suzuka": {
        "overtake_difficulty": 3,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.35,
    },
    # ── China ─────────────────────────────────────────────────────────────────
    "Shanghai": {
        "overtake_difficulty": 2,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.35,
    },
    # ── Miami ─────────────────────────────────────────────────────────────────
    "Miami": {
        "overtake_difficulty": 2,
        "drs_zones": 3,
        "avg_safety_car_prob": 0.45,
    },
    # ── Emilia-Romagna / Imola ────────────────────────────────────────────────
    "Imola": {
        "overtake_difficulty": 4,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.50,
    },
    # ── Monaco ────────────────────────────────────────────────────────────────
    "Monte-Carlo": {
        "overtake_difficulty": 5,
        "drs_zones": 1,
        "avg_safety_car_prob": 0.70,
    },
    # ── Spain / Barcelona ─────────────────────────────────────────────────────
    "Barcelona": {
        "overtake_difficulty": 3,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.20,
    },
    # ── Canada ────────────────────────────────────────────────────────────────
    "Montreal": {
        "overtake_difficulty": 2,
        "drs_zones": 3,
        "avg_safety_car_prob": 0.60,
    },
    # ── Austria ───────────────────────────────────────────────────────────────
    "Spielberg": {
        "overtake_difficulty": 2,
        "drs_zones": 3,
        "avg_safety_car_prob": 0.35,
    },
    # ── Silverstone ───────────────────────────────────────────────────────────
    "Silverstone": {
        "overtake_difficulty": 2,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.40,
    },
    # ── Hungary ───────────────────────────────────────────────────────────────
    "Budapest": {
        "overtake_difficulty": 4,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.25,
    },
    # ── Belgium / Spa ─────────────────────────────────────────────────────────
    "Spa-Francorchamps": {
        "overtake_difficulty": 2,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.50,
    },
    # ── Netherlands / Zandvoort ───────────────────────────────────────────────
    "Zandvoort": {
        "overtake_difficulty": 4,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.30,
    },
    # ── Italy / Monza ─────────────────────────────────────────────────────────
    "Monza": {
        "overtake_difficulty": 1,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.45,
    },
    # ── Azerbaijan / Baku ────────────────────────────────────────────────────
    "Baku": {
        "overtake_difficulty": 2,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.70,
    },
    # ── Singapore ─────────────────────────────────────────────────────────────
    "Singapore": {
        "overtake_difficulty": 4,
        "drs_zones": 3,
        "avg_safety_car_prob": 0.75,
    },
    # ── United States / Austin ────────────────────────────────────────────────
    "Austin": {
        "overtake_difficulty": 2,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.35,
    },
    # ── Mexico City ───────────────────────────────────────────────────────────
    "Mexico City": {
        "overtake_difficulty": 3,
        "drs_zones": 3,
        "avg_safety_car_prob": 0.30,
    },
    # ── Brazil / Interlagos ───────────────────────────────────────────────────
    "São Paulo": {
        "overtake_difficulty": 2,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.60,
    },
    # ── Las Vegas ─────────────────────────────────────────────────────────────
    "Las Vegas": {
        "overtake_difficulty": 2,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.50,
    },
    # ── Qatar / Lusail ────────────────────────────────────────────────────────
    "Lusail": {
        "overtake_difficulty": 2,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.35,
    },
    # ── Abu Dhabi / Yas Marina ────────────────────────────────────────────────
    "Abu Dhabi": {
        "overtake_difficulty": 3,
        "drs_zones": 2,
        "avg_safety_car_prob": 0.25,
    },
}

# Default to use when a circuit is not found in the lookup
DEFAULT_META = {
    "overtake_difficulty": 3,
    "drs_zones": 2,
    "avg_safety_car_prob": 0.40,
}


def get_circuit_meta(location: str) -> dict:
    """
    Returns the metadata dict for *location*.
    Falls back to DEFAULT_META if the circuit is not listed.
    Partial match is attempted (e.g. 'Monte Carlo' → 'Monte-Carlo').
    """
    if location in CIRCUIT_META:
        return CIRCUIT_META[location]

    # Try case-insensitive / hyphen-insensitive match
    norm = location.lower().replace("-", " ").replace("_", " ")
    for key, val in CIRCUIT_META.items():
        if key.lower().replace("-", " ") == norm:
            return val

    return DEFAULT_META.copy()
