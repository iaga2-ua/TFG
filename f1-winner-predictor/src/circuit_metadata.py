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

# --- TABLA ESTATICA ---------------------------------------------------------
# Clave = string Location devuelto por FastF1 (event["Location"])
# track_length_km y corner_count se usan como fallback cuando
# get_circuit_info() falla (sesion cargada sin laps=True).
_STATIC_META: dict[str, dict] = {
    "Sakhir":            {"track_length_km": 5.412, "corner_count": 15, "overtake_difficulty": 1, "drs_zones": 3, "avg_safety_car_prob": 0.25},
    "Jeddah":            {"track_length_km": 6.174, "corner_count": 27, "overtake_difficulty": 2, "drs_zones": 3, "avg_safety_car_prob": 0.55},
    "Melbourne":         {"track_length_km": 5.278, "corner_count": 16, "overtake_difficulty": 2, "drs_zones": 4, "avg_safety_car_prob": 0.60},
    "Suzuka":            {"track_length_km": 5.807, "corner_count": 18, "overtake_difficulty": 3, "drs_zones": 2, "avg_safety_car_prob": 0.35},
    "Shanghai":          {"track_length_km": 5.451, "corner_count": 16, "overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.35},
    "Miami":             {"track_length_km": 5.412, "corner_count": 19, "overtake_difficulty": 2, "drs_zones": 3, "avg_safety_car_prob": 0.45},
    "Imola":             {"track_length_km": 4.909, "corner_count": 19, "overtake_difficulty": 4, "drs_zones": 2, "avg_safety_car_prob": 0.50},
    "Monte-Carlo":       {"track_length_km": 3.337, "corner_count": 19, "overtake_difficulty": 5, "drs_zones": 1, "avg_safety_car_prob": 0.70},
    "Barcelona":         {"track_length_km": 4.657, "corner_count": 14, "overtake_difficulty": 3, "drs_zones": 2, "avg_safety_car_prob": 0.20},
    "Montreal":          {"track_length_km": 4.361, "corner_count": 14, "overtake_difficulty": 2, "drs_zones": 3, "avg_safety_car_prob": 0.60},
    "Spielberg":         {"track_length_km": 4.318, "corner_count": 10, "overtake_difficulty": 2, "drs_zones": 3, "avg_safety_car_prob": 0.35},
    "Silverstone":       {"track_length_km": 5.891, "corner_count": 18, "overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.40},
    "Budapest":          {"track_length_km": 4.381, "corner_count": 14, "overtake_difficulty": 4, "drs_zones": 2, "avg_safety_car_prob": 0.25},
    "Spa-Francorchamps": {"track_length_km": 7.004, "corner_count": 19, "overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.50},
    "Zandvoort":         {"track_length_km": 4.259, "corner_count": 14, "overtake_difficulty": 4, "drs_zones": 2, "avg_safety_car_prob": 0.30},
    "Monza":             {"track_length_km": 5.793, "corner_count": 11, "overtake_difficulty": 1, "drs_zones": 2, "avg_safety_car_prob": 0.45},
    "Baku":              {"track_length_km": 6.003, "corner_count": 20, "overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.70},
    "Singapore":         {"track_length_km": 4.940, "corner_count": 19, "overtake_difficulty": 4, "drs_zones": 3, "avg_safety_car_prob": 0.75},
    "Austin":            {"track_length_km": 5.513, "corner_count": 20, "overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.35},
    "Mexico City":       {"track_length_km": 4.304, "corner_count": 17, "overtake_difficulty": 3, "drs_zones": 3, "avg_safety_car_prob": 0.30},
    "Sao Paulo":         {"track_length_km": 4.309, "corner_count": 15, "overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.60},
    "Las Vegas":         {"track_length_km": 6.201, "corner_count": 17, "overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.50},
    "Lusail":            {"track_length_km": 5.380, "corner_count": 16, "overtake_difficulty": 2, "drs_zones": 2, "avg_safety_car_prob": 0.35},
    "Abu Dhabi":         {"track_length_km": 5.281, "corner_count": 16, "overtake_difficulty": 3, "drs_zones": 2, "avg_safety_car_prob": 0.25},
}

_DEFAULT_STATIC = {
    "track_length_km": float("nan"),
    "corner_count": 0,
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
    # get_circuit_info() requiere laps en memoria; puede fallar si la sesion
    # se cargo con laps=False. En ese caso usamos la tabla estatica como fallback.
    static = _match_static(location)
    corner_count    = static.get("corner_count", 0)
    track_length_km = static.get("track_length_km", float("nan"))
    try:
        ci = session.get_circuit_info()
        if ci is not None:
            # Numero de curvas (FastF1 tiene prioridad sobre la tabla estatica)
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
        logger.debug(
            "circuit_metadata: get_circuit_info() fallo para '%s' (usando tabla estatica): %s",
            location, exc,
        )

    # -- Combinar dinamico + estatico ------------------------------------------
    return {
        "location":             location,
        "country":              country,
        "track_length_km":      track_length_km,
        "corner_count":         corner_count,
        "overtake_difficulty":  static["overtake_difficulty"],
        "drs_zones":            static["drs_zones"],
        "avg_safety_car_prob":  static["avg_safety_car_prob"],
    }


# --- COMPATIBILIDAD CON CODIGO LEGACY ----------------------------------------

def get_circuit_meta(location: str) -> dict:
    """
    Devuelve solo los campos estaticos para un circuito dado su Location.

    Fallback rapido cuando no se dispone de un objeto Session de FastF1,
    por ejemplo en feature_engineering sobre datos ya recopilados.
    Los campos dinamicos (track_length_km, corner_count) NO se incluyen.
    """
    return _match_static(location)
