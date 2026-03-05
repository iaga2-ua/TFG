"""
circuit_metadata.py
===================
Combina datos dinamicos extraidos de la API FastF1 con una tabla estatica
reducida para los campos que FastF1 no proporciona directamente.

Campos dinamicos (FastF1):
    location        (str)   - Nombre corto del circuito  (event["Location"])
    country         (str)   - Pais del GP                (event["Country"])
    track_length_km (float) - Longitud de la pista en km (CircuitInfo)
    corner_count    (int)   - Numero de curvas            (CircuitInfo)

Campos estaticos (tabla interna):
    overtake_difficulty (int, 1-5)
        1 = muy facil (Monza, Bahrein)
        5 = casi imposible (Monaco)
    drs_zones           (int)   - Zonas DRS de activacion
    avg_safety_car_prob (float) - Probabilidad historica de SC/VSC (2018-2025)
"""

import logging

logger = logging.getLogger(__name__)

# --- TABLA ESTATICA (solo lo que FastF1 no ofrece) ---------------------------
# Clave = string Location devuelto por FastF1 (event["Location"])
_STATIC_META: dict[str, dict] = {
    "Sakhir":            {"overtake_difficulty": 1, "drs_zones": 3, "avg_safety_car_prob": 0.25},
    "Jeddah":            {"overtake_difficulty": 2, "drs_zones": 3, "avg_safety_car_prob": 0.55},
    "Melbourne":         {"overtake_difficulty": 2, "drs_zones": 4, "avg_safety_car_prob": 0.60},
    "Suzuka":            {"overtake_difficulty": 3, "drs_zones": 2, "avg_safety_car_prob": 0.35},
    "Shanghai":          {"overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.35},
    "Miami":             {"overtake_difficulty": 2, "drs_zones": 3, "avg_safety_car_prob": 0.45},
    "Imola":             {"overtake_difficulty": 4, "drs_zones": 2, "avg_safety_car_prob": 0.50},
    "Monte-Carlo":       {"overtake_difficulty": 5, "drs_zones": 1, "avg_safety_car_prob": 0.70},
    "Barcelona":         {"overtake_difficulty": 3, "drs_zones": 2, "avg_safety_car_prob": 0.20},
    "Montreal":          {"overtake_difficulty": 2, "drs_zones": 3, "avg_safety_car_prob": 0.60},
    "Spielberg":         {"overtake_difficulty": 2, "drs_zones": 3, "avg_safety_car_prob": 0.35},
    "Silverstone":       {"overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.40},
    "Budapest":          {"overtake_difficulty": 4, "drs_zones": 2, "avg_safety_car_prob": 0.25},
    "Spa-Francorchamps": {"overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.50},
    "Zandvoort":         {"overtake_difficulty": 4, "drs_zones": 2, "avg_safety_car_prob": 0.30},
    "Monza":             {"overtake_difficulty": 1, "drs_zones": 2, "avg_safety_car_prob": 0.45},
    "Baku":              {"overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.70},
    "Singapore":         {"overtake_difficulty": 4, "drs_zones": 3, "avg_safety_car_prob": 0.75},
    "Austin":            {"overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.35},
    "Mexico City":       {"overtake_difficulty": 3, "drs_zones": 3, "avg_safety_car_prob": 0.30},
    "Sao Paulo":         {"overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.60},
    "Las Vegas":         {"overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.50},
    "Lusail":            {"overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.35},
    "Abu Dhabi":         {"overtake_difficulty": 3, "drs_zones": 2, "avg_safety_car_prob": 0.25},
}

_DEFAULT_STATIC = {
    "overtake_difficulty": 3,
    "drs_zones": 2,
    "avg_safety_car_prob": 0.40,
}


def _match_static(location: str) -> dict:
    """Busca en _STATIC_META con tolerancia a mayusculas y guiones."""
    if location in _STATIC_META:
        return _STATIC_META[location].copy()
    norm = location.lower().replace("-", " ").replace("_", " ")
    for key, val in _STATIC_META.items():
        if key.lower().replace("-", " ") == norm:
            return val.copy()
    logger.warning(
        "circuit_metadata: '%s' no encontrado en tabla estatica; usando valores por defecto.",
        location,
    )
    return _DEFAULT_STATIC.copy()


# --- FUNCION PRINCIPAL: extrae de FastF1 -------------------------------------

def get_circuit_meta_from_fastf1(session) -> dict:
    """
    Extrae la metadata del circuito de un objeto Session de FastF1 ya cargado.

    Parametros
    ----------
    session : fastf1.core.Session
        Session ya cargada (basta con session.load(laps=False, ...)).

    Campos devueltos
    ----------------
    location, country       -- del objeto event de FastF1
    track_length_km         -- longitud total de la pista (marshal_sectors)
    corner_count            -- numero de curvas (CircuitInfo.corners)
    overtake_difficulty     -- de la tabla estatica
    drs_zones               -- de la tabla estatica
    avg_safety_car_prob     -- de la tabla estatica
    """
    # -- Datos del evento (location, country) ----------------------------------
    try:
        event    = session.event
        location = str(event.get("Location", "Unknown"))
        country  = str(event.get("Country",  "Unknown"))
    except Exception as exc:
        logger.warning("circuit_metadata: no se pudo leer event info: %s", exc)
        location, country = "Unknown", "Unknown"

    # -- CircuitInfo de FastF1 (track length, corner count) -------------------
    corner_count    = 0
    track_length_km = float("nan")
    try:
        ci = session.get_circuit_info()
        if ci is not None:
            # Numero de curvas
            if hasattr(ci, "corners") and ci.corners is not None:
                corner_count = int(len(ci.corners))

            # Longitud total: distancia acumulada en el ultimo marshal sector
            if (
                hasattr(ci, "marshal_sectors")
                and ci.marshal_sectors is not None
                and not ci.marshal_sectors.empty
            ):
                track_length_m  = float(ci.marshal_sectors["Distance"].max())
                track_length_km = round(track_length_m / 1000, 3)
    except Exception as exc:
        logger.warning(
            "circuit_metadata: get_circuit_info() fallo para '%s': %s", location, exc
        )

    # -- Combinar dinamico + estatico ------------------------------------------
    dynamic = {
        "location":         location,
        "country":          country,
        "track_length_km":  track_length_km,
        "corner_count":     corner_count,
    }
    static = _match_static(location)
    return {**dynamic, **static}


# --- COMPATIBILIDAD CON CODIGO LEGACY ----------------------------------------

def get_circuit_meta(location: str) -> dict:
    """
    Devuelve solo los campos estaticos para un circuito dado su Location.

    Fallback rapido cuando no se dispone de un objeto Session de FastF1,
    por ejemplo en feature_engineering sobre datos ya recopilados.
    Los campos dinamicos (track_length_km, corner_count) NO se incluyen.
    """
    return _match_static(location)
