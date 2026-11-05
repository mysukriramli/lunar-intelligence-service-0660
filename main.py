import os
import sys
import math
import logging
from datetime import datetime, timezone as dt_timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field as SQLField, Session, create_engine, select
import pytz

# Safe import of hijridate with high-precision arithmetic fallback
try:
    from hijridate import Hijri, Gregorian
    HAS_HIJRIDATE = True
except ImportError:
    HAS_HIJRIDATE = False

# ---------------------------------------------------------------------------
# Logging & Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lunar-intelligence-service")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lunar_intelligence.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# ---------------------------------------------------------------------------
# Astronomical Constants & Epochs
# ---------------------------------------------------------------------------
SYNODIC_MONTH = 29.530588853    # Mean synodic period (New Moon to New Moon)
ANOMALISTIC_MONTH = 27.554551   # Perigee to perigee cycle
DRACONIC_MONTH = 27.212220815   # Nodal passage cycle (ecliptic intersection)
ECLIPSE_SEASON_INTERVAL = 173.310038  # Half-eclipse year

# Calibrated reference astronomical epochs (UTC)
REF_NEW_MOON = datetime(2024, 1, 11, 11, 57, 0, tzinfo=dt_timezone.utc)
REF_PERIGEE = datetime(2024, 1, 13, 10, 35, 0, tzinfo=dt_timezone.utc)
REF_ECLIPSE_SEASON = datetime(2024, 4, 8, 18, 18, 0, tzinfo=dt_timezone.utc)

# ---------------------------------------------------------------------------
# Domain Enums
# ---------------------------------------------------------------------------
class StandardizedLunarPhase(str, Enum):
    NEW_MOON = "NEW_MOON"
    WAXING_CRESCENT = "WAXING_CRESCENT"
    FIRST_QUARTER = "FIRST_QUARTER"
    WAXING_GIBBOUS = "WAXING_GIBBOUS"
    FULL_MOON = "FULL_MOON"
    WANING_GIBBOUS = "WANING_GIBBOUS"
    LAST_QUARTER = "LAST_QUARTER"
    WANING_CRESCENT = "WANING_CRESCENT"

class CrescentVisibilityClass(str, Enum):
    IMPOSSIBLE = "IMPOSSIBLE"
    DIFFICULT = "DIFFICULT"
    POSSIBLE = "POSSIBLE"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"

class TidalPotential(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

# ---------------------------------------------------------------------------
# SQLModel Persistence Layer
# ---------------------------------------------------------------------------
class LunarSnapshot(SQLModel, table=True):
    __tablename__ = "lunar_snapshots"
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    timestamp_utc: str = SQLField(index=True)
    phase_name: str
    illumination_percent: float
    moon_age_days: float
    waxing: bool
    next_new_moon: str
    next_full_moon: str
    hijri_date: str
    lunar_month: str
    tidal_classification: str
    illumination_classification: str
    standardized_phase: Optional[str] = SQLField(default=None)
    lunar_state_index: Optional[float] = SQLField(default=None)
    crescent_visibility: Optional[str] = SQLField(default=None)
    tidal_potential: Optional[str] = SQLField(default=None)
    created_at: str = SQLField(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat())

def init_db():
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database initialized successfully.")
    except Exception as exc:
        logger.warning(f"Database initialization warning: {exc}")

# ---------------------------------------------------------------------------
# FastAPI Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="lunar-intelligence-service-0660",
    description="Autonomous Moon Phase, Hijri Calendar, Tide & Environmental Signal Agent",
    version="2.0.0"
)

@app.on_event("startup")
def on_startup():
    init_db()

# ---------------------------------------------------------------------------
# Astronomical & Psychrometric Computation Engines
# ---------------------------------------------------------------------------
def calculate_moon_age(target_dt: Optional[datetime] = None) -> float:
    """
    Step 1: Moon Age Engine.
    Computes precise fractional moon age in days (0.0 to 29.53) using synodic cycles.
    """
    if target_dt is None:
        target_dt = datetime.now(dt_timezone.utc)
    elif target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=dt_timezone.utc)
    else:
        target_dt = target_dt.astimezone(dt_timezone.utc)

    diff_days = (target_dt - REF_NEW_MOON).total_seconds() / 86400.0
    moon_age = diff_days % SYNODIC_MONTH
    if moon_age < 0:
        moon_age += SYNODIC_MONTH
    return float(moon_age)

def calculate_lunar_illumination(moon_age: float) -> float:
    """
    Step 2: Lunar Illumination Engine.
    Computes disk illumination percent (0.0 - 100.0%) from lunar phase angle phi:
    k = (1 - cos(phi)) / 2 * 100
    """
    phi = 2.0 * math.pi * (moon_age / SYNODIC_MONTH)
    illum = (1.0 - math.cos(phi)) / 2.0 * 100.0
    return float(round(max(0.0, min(100.0, illum)), 2))

def classify_lunar_phase(moon_age: float) -> Tuple[StandardizedLunarPhase, str]:
    """
    Step 3: Lunar Phase Classification.
    Standardized 8-phase sectoring centered on primary and intermediate celestial positions.
    Returns (StandardizedLunarPhase, DisplayString).
    """
    step = SYNODIC_MONTH / 8.0
    half_step = step / 2.0  # ~1.84566 days

    if moon_age < half_step:
        return StandardizedLunarPhase.NEW_MOON, "New Moon"
    elif moon_age < half_step + step:
        return StandardizedLunarPhase.WAXING_CRESCENT, "Waxing Crescent"
    elif moon_age < half_step + 2 * step:
        return StandardizedLunarPhase.FIRST_QUARTER, "First Quarter"
    elif moon_age < half_step + 3 * step:
        return StandardizedLunarPhase.WAXING_GIBBOUS, "Waxing Gibbous"
    elif moon_age < half_step + 4 * step:
        return StandardizedLunarPhase.FULL_MOON, "Full Moon"
    elif moon_age < half_step + 5 * step:
        return StandardizedLunarPhase.WANING_GIBBOUS, "Waning Gibbous"
    elif moon_age < half_step + 6 * step:
        return StandardizedLunarPhase.LAST_QUARTER, "Last Quarter"
    elif moon_age < half_step + 7 * step:
        return StandardizedLunarPhase.WANING_CRESCENT, "Waning Crescent"
    else:
        return StandardizedLunarPhase.NEW_MOON, "New Moon"

def predict_next_full_moon(target_dt: datetime, moon_age: float) -> Dict[str, Any]:
    """
    Step 4: Full Moon Prediction Engine.
    Dynamically projects the next Full Moon epoch (age = SYNODIC_MONTH / 2.0).
    """
    full_moon_target = SYNODIC_MONTH / 2.0
    if moon_age < full_moon_target:
        days_remaining = full_moon_target - moon_age
    else:
        days_remaining = (SYNODIC_MONTH - moon_age) + full_moon_target

    next_dt = target_dt + timedelta(days=days_remaining)
    return {
        "datetime": next_dt.isoformat(),
        "days_remaining": round(days_remaining, 2)
    }

def predict_next_new_moon(target_dt: datetime, moon_age: float) -> Dict[str, Any]:
    """
    Step 5: New Moon Prediction Engine.
    Dynamically projects the next New Moon astronomical syzygy.
    """
    days_remaining = SYNODIC_MONTH - moon_age
    if days_remaining <= 0:
        days_remaining += SYNODIC_MONTH

    next_dt = target_dt + timedelta(days=days_remaining)
    return {
        "datetime": next_dt.isoformat(),
        "days_remaining": round(days_remaining, 2)
    }

def _fallback_gregorian_to_hijri(year: int, month: int, day: int) -> Tuple[int, int, int]:
    """Astronomical algorithmic conversion from Gregorian to Islamic Hijri date."""
    if month < 3:
        year -= 1
        month += 12
    a = math.floor(year / 100.0)
    b = 2 - a + math.floor(a / 4.0)
    jd = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b - 1524.5
    
    epoch_jd = 1948439.5
    days_since_epoch = jd - epoch_jd
    h_year = math.floor((30 * days_since_epoch + 10646) / 10631)
    
    start_of_year_jd = epoch_jd + math.floor((10631 * (h_year - 1) + 301) / 30)
    day_of_year = jd - start_of_year_jd + 1
    
    h_month = min(12, math.ceil((day_of_year - 0.5) / 29.5))
    start_of_month_jd = epoch_jd + math.floor((10631 * (h_year - 1) + 301) / 30) + math.ceil(29.5 * (h_month - 1))
    h_day = int(jd - start_of_month_jd + 1)
    if h_day < 1:
        h_month -= 1
        if h_month < 1:
            h_month = 12
            h_year -= 1
        h_day = 30
    elif h_day > 30:
        h_day = 1
        h_month += 1
        if h_month > 12:
            h_month = 1
            h_year += 1
    return int(h_year), int(h_month), int(h_day)

def compute_hijri_intelligence(target_dt: datetime) -> Dict[str, Any]:
    """
    Step 6: Hijri Intelligence Layer.
    Supplies machine-consumable boolean and categorical calendar markers.
    """
    greg_date = target_dt.date()
    h_year, h_month, h_day = (1448, 1, 1)

    if HAS_HIJRIDATE:
        try:
            h_obj = Gregorian(greg_date.year, greg_date.month, greg_date.day).to_hijri()
            h_year, h_month, h_day = h_obj.year, h_obj.month, h_obj.day
        except Exception:
            h_year, h_month, h_day = _fallback_gregorian_to_hijri(greg_date.year, greg_date.month, greg_date.day)
    else:
        h_year, h_month, h_day = _fallback_gregorian_to_hijri(greg_date.year, greg_date.month, greg_date.day)

    hijri_months = [
        "Muharram", "Safar", "Rabi' al-Awwal", "Rabi' al-Thani",
        "Jumada al-Ula", "Jumada al-Thani", "Rajab", "Sha'ban",
        "Ramadan", "Shawwal", "Dhu al-Qi'dah", "Dhu al-Hijjah"
    ]
    month_name = hijri_months[h_month - 1] if 1 <= h_month <= 12 else "Unknown"
    date_str = f"{h_year}-{h_month:02d}-{h_day:02d}"

    is_ramadan = (h_month == 9)
    is_dhul_hijjah = (h_month == 12)
    is_ashura = (h_month == 1 and h_day == 10)
    is_laylat_al_qadr_window = (h_month == 9 and h_day >= 20)
    is_white_days = (h_day in [13, 14, 15])
    is_eid_al_fitr = (h_month == 10 and h_day in [1, 2, 3])
    is_eid_al_adha = (h_month == 12 and h_day in [10, 11, 12, 13])
    is_arafah = (h_month == 12 and h_day == 9)

    # Hierarchical religious / astronomical significance score (0 - 100)
    if is_laylat_al_qadr_window:
        significance_score = 100.0
    elif is_arafah or is_eid_al_fitr or is_eid_al_adha:
        significance_score = 95.0
    elif is_ramadan or (is_dhul_hijjah and h_day <= 10):
        significance_score = 85.0
    elif is_ashura:
        significance_score = 80.0
    elif is_white_days:
        significance_score = 65.0
    elif h_month in [1, 7, 11, 12]:
        significance_score = 45.0
    else:
        significance_score = 20.0

    return {
        "current_hijri_date": date_str,
        "hijri_year": h_year,
        "hijri_month": h_month,
        "hijri_day": h_day,
        "hijri_month_name": month_name,
        "is_ramadan": is_ramadan,
        "is_dhul_hijjah": is_dhul_hijjah,
        "is_ashura": is_ashura,
        "is_laylat_al_qadr_window": is_laylat_al_qadr_window,
        "is_white_days": is_white_days,
        "is_eid_al_fitr": is_eid_al_fitr,
        "is_eid_al_adha": is_eid_al_adha,
        "significance_score": significance_score
    }

def calculate_crescent_visibility(
    moon_age: float,
    illumination: float,
    elongation: Optional[float] = None,
    sunset_offset: float = 30.0
) -> Dict[str, Any]:
    """
    Step 7: Crescent Visibility Engine (Hilal Sighting).
    Uses Yallop / Danjon criteria (elongation, Danjon limit ~7.0 deg, sunset lag).
    """
    if elongation is None:
        frac = (moon_age % SYNODIC_MONTH) / SYNODIC_MONTH
        angle = frac * 360.0
        elongation = min(angle, 360.0 - angle)

    lag_factor = max(0.1, min(1.0, sunset_offset / 45.0))

    # Danjon optical limit: Crescent physically invisible if elongation < 7 deg or age < 14h
    if elongation < 7.0 or moon_age < 0.6 or illumination < 0.4:
        score = 0.0
        vis_class = CrescentVisibilityClass.IMPOSSIBLE.value
    elif elongation < 9.0 or moon_age < 0.85:
        score = max(5.0, min(35.0, 15.0 + (elongation - 7.0) * 10.0 * lag_factor))
        vis_class = CrescentVisibilityClass.DIFFICULT.value
    elif elongation < 11.5 or moon_age < 1.3:
        score = max(35.0, min(65.0, 35.0 + (elongation - 9.0) * 12.0 * lag_factor))
        vis_class = CrescentVisibilityClass.POSSIBLE.value
    elif elongation < 15.0 or moon_age < 2.2:
        score = max(65.0, min(85.0, 65.0 + (elongation - 11.5) * 6.0 * lag_factor))
        vis_class = CrescentVisibilityClass.GOOD.value
    else:
        score = max(85.0, min(100.0, 85.0 + (elongation - 15.0) * 1.0 * lag_factor))
        vis_class = CrescentVisibilityClass.EXCELLENT.value

    return {
        "moon_age_days": round(moon_age, 2),
        "illumination_percent": round(illumination, 2),
        "elongation_degrees": round(elongation, 2),
        "sunset_offset_minutes": round(sunset_offset, 1),
        "visibility_score": round(score, 1),
        "visibility_class": vis_class
    }

def calculate_tidal_potential(target_dt: datetime, moon_age: float) -> Dict[str, Any]:
    """
    Step 8: Tidal Potential Index.
    Integrates syzygy alignment and anomalistic perigee cycles to determine marine tidal force.
    """
    days_to_new = min(moon_age, SYNODIC_MONTH - moon_age)
    days_to_full = abs(moon_age - (SYNODIC_MONTH / 2.0))
    syzygy_dist = min(days_to_new, days_to_full)

    days_to_q1 = abs(moon_age - (SYNODIC_MONTH / 4.0))
    days_to_q3 = abs(moon_age - (3.0 * SYNODIC_MONTH / 4.0))
    quad_dist = min(days_to_q1, days_to_q3)

    # Anomalistic month perigee proximity
    diff_perigee = (target_dt - REF_PERIGEE).total_seconds() / 86400.0
    anom_phase = diff_perigee % ANOMALISTIC_MONTH
    dist_to_perigee = min(anom_phase, ANOMALISTIC_MONTH - anom_phase)

    # Gravitational tidal forcing coefficient (standard 20 - 120 scale)
    phase_factor = math.cos(4.0 * math.pi * (moon_age / SYNODIC_MONTH))
    perigee_factor = math.cos(2.0 * math.pi * (anom_phase / ANOMALISTIC_MONTH))
    tide_coef = max(20.0, min(120.0, 70.0 + 32.0 * phase_factor + 16.0 * perigee_factor))

    if tide_coef >= 100.0 or (syzygy_dist <= 1.5 and dist_to_perigee <= 2.2):
        potential = TidalPotential.EXTREME.value
        legacy_class = "SPRING_TIDE"
    elif tide_coef >= 78.0 or syzygy_dist <= 2.2:
        potential = TidalPotential.HIGH.value
        legacy_class = "SPRING_TIDE"
    elif tide_coef >= 55.0 and quad_dist > 2.0:
        potential = TidalPotential.MODERATE.value
        legacy_class = "TRANSITIONAL"
    else:
        potential = TidalPotential.LOW.value
        legacy_class = "NEAP_TIDE"

    return {
        "tidal_potential": potential,
        "legacy_tidal_classification": legacy_class,
        "tidal_coefficient": round(tide_coef, 1),
        "syzygy_distance_days": round(syzygy_dist, 2),
        "perigee_distance_days": round(dist_to_perigee, 2)
    }

def calculate_eclipse_window(target_dt: datetime, moon_age: float) -> Dict[str, Any]:
    """
    Step 9: Eclipse Awareness Layer.
    Calculates proximity to the nearest eclipse window and nodal intersection.
    """
    diff_season_days = (target_dt - REF_ECLIPSE_SEASON).total_seconds() / 86400.0
    season_offset = diff_season_days % ECLIPSE_SEASON_INTERVAL
    dist_to_season_center = min(season_offset, ECLIPSE_SEASON_INTERVAL - season_offset)

    is_in_eclipse_season = dist_to_season_center <= 17.5
    days_to_window = 0.0 if is_in_eclipse_season else dist_to_season_center - 17.5

    # Draconic nodal alignment
    half_draconic = DRACONIC_MONTH / 2.0
    node_phase = diff_season_days % half_draconic
    dist_to_node = min(node_phase, half_draconic - node_phase)

    days_to_new = min(moon_age, SYNODIC_MONTH - moon_age)
    days_to_full = abs(moon_age - (SYNODIC_MONTH / 2.0))
    syzygy_dist = min(days_to_new, days_to_full)

    if is_in_eclipse_season and syzygy_dist <= 1.3 and dist_to_node <= 1.6:
        status = "ECLIPSE_IMMINENT"
        eclipse_type = "SOLAR_ECLIPSE" if days_to_new < days_to_full else "LUNAR_ECLIPSE"
    elif is_in_eclipse_season:
        status = "INSIDE_ECLIPSE_SEASON"
        eclipse_type = "POTENTIAL_SYZYGY_WINDOW"
    else:
        status = "OUTSIDE_ECLIPSE_SEASON"
        eclipse_type = "NONE"

    return {
        "days_to_possible_eclipse_window": round(days_to_window, 1),
        "is_in_eclipse_season": is_in_eclipse_season,
        "days_to_season_center": round(dist_to_season_center, 1),
        "node_distance_days": round(dist_to_node, 2),
        "syzygy_distance_days": round(syzygy_dist, 2),
        "eclipse_status": status,
        "eclipse_type": eclipse_type
    }

def compute_lunar_state_index(
    illumination: float,
    tidal_potential: str,
    visibility_score: float,
    hijri_sig_score: float
) -> float:
    """
    Step 10: Unified Lunar Signal.
    Synthesizes optical illumination, gravitational tidal pull, crescent observation viability,
    and calendar significance into a normalized 0 - 100 indicator.
    """
    tide_weight = {
        TidalPotential.LOW.value: 25.0,
        TidalPotential.MODERATE.value: 50.0,
        TidalPotential.HIGH.value: 75.0,
        TidalPotential.EXTREME.value: 100.0
    }.get(tidal_potential, 50.0)

    raw_index = (
        0.35 * illumination +
        0.25 * tide_weight +
        0.20 * visibility_score +
        0.20 * hijri_sig_score
    )
    return float(round(max(0.0, min(100.0, raw_index)), 2))

# ---------------------------------------------------------------------------
# Comprehensive Master Orchestrator
# ---------------------------------------------------------------------------
def compute_lunar_data(target_dt: datetime) -> Dict[str, Any]:
    """Assembles all lunar, psychrometric, tidal, and religious intelligence layers."""
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=dt_timezone.utc)
    else:
        target_dt = target_dt.astimezone(dt_timezone.utc)

    moon_age = calculate_moon_age(target_dt)
    illumination = calculate_lunar_illumination(moon_age)
    std_phase, display_phase = classify_lunar_phase(moon_age)
    waxing = moon_age < (SYNODIC_MONTH / 2.0)

    # Illumination categorical classification
    if illumination <= 10.0:
        illum_class = "DARK"
    elif illumination <= 35.0:
        illum_class = "LOW"
    elif illumination <= 75.0:
        illum_class = "MEDIUM"
    else:
        illum_class = "BRIGHT"

    next_full = predict_next_full_moon(target_dt, moon_age)
    next_new = predict_next_new_moon(target_dt, moon_age)
    hijri_info = compute_hijri_intelligence(target_dt)
    crescent_vis = calculate_crescent_visibility(moon_age, illumination)
    tidal_info = calculate_tidal_potential(target_dt, moon_age)
    eclipse_info = calculate_eclipse_window(target_dt, moon_age)

    lunar_index = compute_lunar_state_index(
        illumination=illumination,
        tidal_potential=tidal_info["tidal_potential"],
        visibility_score=crescent_vis["visibility_score"],
        hijri_sig_score=hijri_info["significance_score"]
    )

    return {
        "timestamp_utc": target_dt.isoformat(),
        "phase_name": std_phase.value,
        "phase": std_phase.value,
        "phase_display": display_phase,
        "standardized_phase": std_phase.value,
        "illumination_percent": illumination,
        "illumination": illumination,
        "moon_age_days": round(moon_age, 2),
        "waxing": waxing,
        "next_new_moon": next_new["datetime"],
        "next_full_moon": next_full["datetime"],
        "next_new_moon_prediction": next_new,
        "next_full_moon_prediction": next_full,
        "hijri_date": hijri_info["current_hijri_date"],
        "current_hijri_date": hijri_info["current_hijri_date"],
        "lunar_month": hijri_info["hijri_month_name"],
        "hijri_month_name": hijri_info["hijri_month_name"],
        "is_ramadan": hijri_info["is_ramadan"],
        "is_dhul_hijjah": hijri_info["is_dhul_hijjah"],
        "is_ashura": hijri_info["is_ashura"],
        "is_laylat_al_qadr_window": hijri_info["is_laylat_al_qadr_window"],
        "tidal_classification": tidal_info["legacy_tidal_classification"],
        "tidal_potential": tidal_info["tidal_potential"],
        "tidal_details": tidal_info,
        "illumination_classification": illum_class,
        "crescent_visibility": crescent_vis,
        "eclipse_window": eclipse_info,
        "lunar_state_index": lunar_index
    }

def convert_timezone(dt_str: str, tz_name: str) -> str:
    try:
        dt = datetime.fromisoformat(dt_str)
        target_tz = pytz.timezone(tz_name)
        return dt.astimezone(target_tz).isoformat()
    except Exception:
        return dt_str

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "status_code": "healthy",
        "service": "lunar-intelligence-service-0660"
    }

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy"}

@app.get("/heartbeat")
def heartbeat() -> Dict[str, Any]:
    return {
        "status": "alive",
        "timestamp_utc": datetime.now(dt_timezone.utc).isoformat(),
        "service": "lunar-intelligence-service-0660"
    }

@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "get_lunar_status",
                "description": "Returns current lunar operational intelligence including phase, illumination, tidal potential, and crescent visibility.",
                "input_schema": {"lat": "float", "lon": "float", "timezone": "string"}
            },
            {
                "name": "predict_hijri_events",
                "description": "Returns predicted Islamic calendar events calculated from lunar cycle models.",
                "input_schema": {"year": "int", "method": "string"}
            },
            {
                "name": "forecast_lunar_conditions",
                "description": "Forecast future lunar lighting, tidal potential, and visibility states.",
                "input_schema": {"days": "int"}
            },
            {
                "name": "get_agent_triggers",
                "description": "Returns machine-actionable triggers for astronomical, marine, and religious agents.",
                "input_schema": {"days": "int"}
            },
            {
                "name": "calculate_crescent_visibility",
                "description": "Calculates astronomical crescent visibility and Hilal sighting probability.",
                "input_schema": {"moon_age": "float", "illumination": "float", "sunset_offset": "float"}
            },
            {
                "name": "get_unified_lunar_signal",
                "description": "Computes composite lunar_state_index (0-100) for autonomous decision-making.",
                "input_schema": {"target_date": "string"}
            }
        ]
    }

@app.get("/v1/moon/current")
def get_current_moon(
    timezone: Optional[str] = Query(None, description="IANA timezone name e.g. Asia/Kuala_Lumpur")
) -> Dict[str, Any]:
    now = datetime.now(dt_timezone.utc)
    data = compute_lunar_data(now)
    if timezone:
        try:
            pytz.timezone(timezone)
            data["timestamp_utc"] = convert_timezone(data["timestamp_utc"], timezone)
            data["next_new_moon"] = convert_timezone(data["next_new_moon"], timezone)
            data["next_full_moon"] = convert_timezone(data["next_full_moon"], timezone)
            data["timezone"] = timezone
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid IANA timezone: {timezone}")
    return data

@app.get("/v1/moon/date/{date}")
def get_moon_by_date(
    date: str = Path(..., description="Date in YYYY-MM-DD format"),
    timezone: Optional[str] = Query(None, description="IANA timezone name")
) -> Dict[str, Any]:
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=dt_timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    data = compute_lunar_data(parsed_date)
    if timezone:
        try:
            pytz.timezone(timezone)
            data["timestamp_utc"] = convert_timezone(data["timestamp_utc"], timezone)
            data["next_new_moon"] = convert_timezone(data["next_new_moon"], timezone)
            data["next_full_moon"] = convert_timezone(data["next_full_moon"], timezone)
            data["timezone"] = timezone
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid IANA timezone: {timezone}")
    return data

@app.get("/v1/moon/phases")
def get_moon_phases() -> Dict[str, Any]:
    return {
        "phases": [
            {"name": "New Moon", "standardized_phase": StandardizedLunarPhase.NEW_MOON.value, "illumination_range": "0-10%"},
            {"name": "Waxing Crescent", "standardized_phase": StandardizedLunarPhase.WAXING_CRESCENT.value, "illumination_range": "10-35%"},
            {"name": "First Quarter", "standardized_phase": StandardizedLunarPhase.FIRST_QUARTER.value, "illumination_range": "35-50%"},
            {"name": "Waxing Gibbous", "standardized_phase": StandardizedLunarPhase.WAXING_GIBBOUS.value, "illumination_range": "50-75%"},
            {"name": "Full Moon", "standardized_phase": StandardizedLunarPhase.FULL_MOON.value, "illumination_range": "75-100%"},
            {"name": "Waning Gibbous", "standardized_phase": StandardizedLunarPhase.WANING_GIBBOUS.value, "illumination_range": "50-75%"},
            {"name": "Last Quarter", "standardized_phase": StandardizedLunarPhase.LAST_QUARTER.value, "illumination_range": "35-50%"},
            {"name": "Waning Crescent", "standardized_phase": StandardizedLunarPhase.WANING_CRESCENT.value, "illumination_range": "10-35%"}
        ]
    }

def _estimate_hijri_events_for_year(greg_year: int) -> Dict[str, str]:
    """Calculates canonical Islamic events using forward calendar scans."""
    events: Dict[str, str] = {}
    cur = datetime(greg_year, 1, 1, tzinfo=dt_timezone.utc)
    end = datetime(greg_year, 12, 31, tzinfo=dt_timezone.utc)
    
    ramadan_start = None
    eid_al_fitr = None
    eid_al_adha = None
    
    d = cur
    while d <= end:
        h_info = compute_hijri_intelligence(d)
        if h_info["hijri_month"] == 9 and h_info["hijri_day"] == 1 and not ramadan_start:
            ramadan_start = d.strftime("%Y-%m-%d")
        if h_info["hijri_month"] == 10 and h_info["hijri_day"] == 1 and not eid_al_fitr:
            eid_al_fitr = d.strftime("%Y-%m-%d")
        if h_info["hijri_month"] == 12 and h_info["hijri_day"] == 10 and not eid_al_adha:
            eid_al_adha = d.strftime("%Y-%m-%d")
        d += timedelta(days=1)
        
    events["ramadan_start"] = ramadan_start or f"{greg_year}-03-01"
    events["eid_al_fitr"] = eid_al_fitr or f"{greg_year}-03-31"
    events["eid_al_adha"] = eid_al_adha or f"{greg_year}-06-06"
    return events

@app.get("/v1/moon/hijri")
def get_hijri_info(
    year: Optional[int] = Query(None, description="Gregorian year"),
    method: str = Query("umm_al_qura", description="Calculation method")
) -> Dict[str, Any]:
    target_date = datetime.now(dt_timezone.utc)
    if year:
        target_date = datetime(year, 6, 1, tzinfo=dt_timezone.utc)
    
    data = compute_lunar_data(target_date)
    estimated_events = _estimate_hijri_events_for_year(target_date.year)
    
    return {
        "method": method,
        "current_hijri_date": data["hijri_date"],
        "lunar_month": data["lunar_month"],
        "hijri_month_name": data["hijri_month_name"],
        "is_ramadan": data["is_ramadan"],
        "is_dhul_hijjah": data["is_dhul_hijjah"],
        "is_ashura": data["is_ashura"],
        "is_laylat_al_qadr_window": data["is_laylat_al_qadr_window"],
        "estimated_events": estimated_events,
        "white_days": [13, 14, 15]
    }

@app.get("/v1/moon/events")
def get_moon_events(days: int = Query(14, description="Lookahead days")) -> Dict[str, Any]:
    events = []
    now = datetime.now(dt_timezone.utc)
    for i in range(days):
        check_dt = now + timedelta(days=i)
        data = compute_lunar_data(check_dt)
        phase_str = data["phase_name"].upper()

        if "FULL" in phase_str:
            events.append({
                "type": "moon.full_moon.approaching",
                "severity": "info",
                "timestamp": check_dt.isoformat(),
                "details": data
            })
        elif "NEW" in phase_str:
            events.append({
                "type": "moon.new_moon.approaching",
                "severity": "info",
                "timestamp": check_dt.isoformat(),
                "details": data
            })
        if data["tidal_potential"] in ["HIGH", "EXTREME"] or data["tidal_classification"] == "SPRING_TIDE":
            events.append({
                "type": "moon.spring_tide.warning",
                "severity": "warning",
                "timestamp": check_dt.isoformat(),
                "details": data
            })
        if data["is_ramadan"]:
            events.append({
                "type": "calendar.ramadan.active",
                "severity": "info",
                "timestamp": check_dt.isoformat(),
                "details": {"hijri_date": data["hijri_date"]}
            })
    return {"events": events}

@app.get("/v1/moon/forecast")
def get_moon_forecast(days: int = Query(30, description="Forecast days")) -> Dict[str, Any]:
    forecast = []
    now = datetime.now(dt_timezone.utc)
    for i in range(days):
        target = now + timedelta(days=i)
        data = compute_lunar_data(target)
        forecast.append({
            "date": target.strftime("%Y-%m-%d"),
            "phase": data["phase_name"],
            "standardized_phase": data["standardized_phase"],
            "illumination": data["illumination_percent"],
            "tidal_classification": data["tidal_classification"],
            "tidal_potential": data["tidal_potential"],
            "illumination_classification": data["illumination_classification"],
            "lunar_state_index": data["lunar_state_index"]
        })
    return {"forecast": forecast}

# ---------------------------------------------------------------------------
# MCP Tool Endpoints
# ---------------------------------------------------------------------------
class LunarStatusInput(BaseModel):
    lat: float = Field(default=3.139)
    lon: float = Field(default=101.6869)
    timezone: str = "UTC"

@app.post("/mcp/get_lunar_status")
def mcp_get_lunar_status(payload: LunarStatusInput) -> Dict[str, Any]:
    if not (-90.0 <= payload.lat <= 90.0) or not (-180.0 <= payload.lon <= 180.0):
        raise HTTPException(status_code=400, detail="Invalid coordinates: lat [-90, 90], lon [-180, 180]")
    
    data = compute_lunar_data(datetime.now(dt_timezone.utc))
    return {
        "phase": data["phase_name"],
        "standardized_phase": data["standardized_phase"],
        "illumination": data["illumination_percent"],
        "moon_age_days": data["moon_age_days"],
        "tidal_classification": data["tidal_classification"],
        "tidal_potential": data["tidal_potential"],
        "illumination_classification": data["illumination_classification"],
        "next_full_moon": convert_timezone(data["next_full_moon"], payload.timezone),
        "next_new_moon": convert_timezone(data["next_new_moon"], payload.timezone),
        "lunar_state_index": data["lunar_state_index"],
        "crescent_visibility": data["crescent_visibility"]
    }

class PredictHijriInput(BaseModel):
    year: int = Field(default=2027)
    method: str = "umm_al_qura"

@app.post("/mcp/predict_hijri_events")
def mcp_predict_hijri(payload: PredictHijriInput) -> Dict[str, Any]:
    events = _estimate_hijri_events_for_year(payload.year)
    return {
        "year": payload.year,
        "method": payload.method,
        "ramadan_start": events["ramadan_start"],
        "eid_al_fitr": events["eid_al_fitr"],
        "eid_al_adha": events["eid_al_adha"]
    }

class ForecastInput(BaseModel):
    days: int = Field(default=30, ge=1, le=180)

@app.post("/mcp/forecast_lunar_conditions")
def mcp_forecast(payload: ForecastInput) -> Dict[str, Any]:
    return get_moon_forecast(days=payload.days)

class TriggersInput(BaseModel):
    days: int = Field(default=14, ge=1, le=90)

@app.post("/mcp/get_agent_triggers")
def mcp_triggers(payload: TriggersInput) -> Dict[str, Any]:
    return get_moon_events(days=payload.days)

# ---------------------------------------------------------------------------
# CLI & Standalone Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
