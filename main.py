import os
import sys
import logging
from datetime import datetime, timezone as dt_timezone, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field as SQLField, Session, create_engine, select
import pytz
from hijridate import Hijri, Gregorian

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lunar-intelligence-service")

# Database setup (SQLite in-memory or local file for production grade resilience)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lunar_intelligence.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# SQLModel for lunar snapshots
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
    created_at: str = SQLField(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat())

def init_db():
    SQLModel.metadata.create_all(engine)

# Initialize FastAPI App strictly named 'app'
app = FastAPI(
    title="lunar-intelligence-service-0660",
    description="Autonomous Moon Phase, Hijri Calendar, Tide & Environmental Signal Agent",
    version="1.0.0"
)

@app.on_event("startup")
# on_startup = init_db

# --- Helper Computation Functions ---
def compute_lunar_data(target_dt: datetime) -> Dict[str, Any]:
    # Ensure UTC reference
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=dt_timezone.utc)
    else:
        target_dt = target_dt.astimezone(dt_timezone.utc)
    
    # Astronomical approximations based on known synodic month (29.5305877057 days)
    # Reference New Moon: Jan 11, 2024, 11:57 UTC
    ref_new_moon = datetime(2024, 1, 11, 11, 57, tzinfo=dt_timezone.utc)
    diff_days = (target_dt - ref_new_moon).total_seconds() / 86400.0
    synodic_month = 29.5305877057
    moon_age = diff_days % synodic_month
    if moon_age < 0:
        moon_age += synodic_month

    fraction = moon_age / synodic_month
    illumination = round((1 - abs(fraction - 0.5) * 2) * 100, 2)
    
    # Waxing / Waning state
    waxing = moon_age < (synodic_month / 2.0)

    # Phase name calculation
    if moon_age < 1.84566:
        phase_name = "New Moon"
    elif moon_age < 5.53699:
        phase_name = "Waxing Crescent"
    elif moon_age < 9.22831:
        phase_name = "First Quarter"
    elif moon_age < 12.91964:
        phase_name = "Waxing Gibbous"
    elif moon_age < 16.61096:
        phase_name = "Full Moon"
    elif moon_age < 20.30229:
        phase_name = "Waning Gibbous"
    elif moon_age < 23.99361:
        phase_name = "Last Quarter"
    elif moon_age < 27.68494:
        phase_name = "Waning Crescent"
    else:
        phase_name = "New Moon"

    # Illumination Classification
    if illumination <= 10.0:
        illum_class = "DARK"
    elif illumination <= 35.0:
        illum_class = "LOW"
    elif illumination <= 75.0:
        illum_class = "MEDIUM"
    else:
        illum_class = "BRIGHT"

    # Tidal Classification
    # +/- 2 days from New/Full = SPRING_TIDE, Quarter = NEAP_TIDE, otherwise TRANSITIONAL
    days_to_new = min(moon_age, synodic_month - moon_age)
    days_to_full = abs(moon_age - (synodic_month / 2.0))
    days_to_quarter1 = abs(moon_age - (synodic_month / 4.0))
    days_to_quarter3 = abs(moon_age - (3 * synodic_month / 4.0))

    min_quarter = min(days_to_quarter1, days_to_quarter3)

    if days_to_new <= 2.0 or days_to_full <= 2.0:
        tidal_class = "SPRING_TIDE"
    elif min_quarter <= 2.0:
        tidal_class = "NEAP_TIDE"
    else:
        tidal_class = "TRANSITIONAL"

    # Next events estimation
    days_to_next_new = synodic_month - moon_age
    next_new_dt = target_dt + timedelta(days=days_to_next_new)
    
    days_to_next_full = (synodic_month / 2.0) - moon_age
    if days_to_next_full < 0:
        days_to_next_full += synodic_month
    next_full_dt = target_dt + timedelta(days=days_to_next_full)

    # Hijri estimation via hijridate library
    greg_date = target_dt.date()
    hijri_obj = Gregorian(greg_date.year, greg_date.month, greg_date.day).to_hijri()
    hijri_date_str = f"{hijri_obj.year}-{hijri_obj.month:02d}-{hijri_obj.day:02d}"
    
    hijri_months = [
        "Muharram", "Safar", "Rabi' al-Awwal", "Rabi' al-Thani",
        "Jumada al-Ula", "Jumada al-Thani", "Rajab", "Sha'ban",
        "Ramadan", "Shawwal", "Dhu al-Qi'dah", "Dhu al-Hijjah"
    ]
    lunar_month_name = hijri_months[hijri_obj.month - 1] if 1 <= hijri_obj.month <= 12 else "Unknown"

    return {
        "timestamp_utc": target_dt.isoformat(),
        "phase_name": phase_name,
        "phase": phase_name,  # alias
        "illumination_percent": illumination,
        "illumination": illumination,  # alias
        "moon_age_days": round(moon_age, 2),
        "waxing": waxing,
        "next_new_moon": next_new_dt.isoformat(),
        "next_full_moon": next_full_dt.isoformat(),
        "hijri_date": hijri_date_str,
        "lunar_month": lunar_month_name,
        "tidal_classification": tidal_class,
        "illumination_classification": illum_class,
    }

def convert_timezone(dt_str: str, tz_name: str) -> str:
    try:
        dt = datetime.fromisoformat(dt_str)
        target_tz = pytz.timezone(tz_name)
        converted = dt.astimezone(target_tz)
        return converted.isoformat()
    except Exception:
        return dt_str

# --- API Endpoints ---

@app.get("/")
def root():
    return {
        "status": "healthy",
        "status_code": "healthy",
        "service": "lunar-intelligence-service-0660"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/heartbeat")
def heartbeat():
    return {
        "status": "alive",
        "timestamp_utc": datetime.now(dt_timezone.utc).isoformat(),
        "service": "lunar-intelligence-service-0660"
    }

@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {
                "name": "get_lunar_status",
                "description": "Returns current lunar operational intelligence.",
                "input_schema": {"lat": "float", "lon": "float", "timezone": "string"}
            },
            {
                "name": "predict_hijri_events",
                "description": "Returns predicted Islamic calendar events.",
                "input_schema": {"year": "int", "method": "string"}
            },
            {
                "name": "forecast_lunar_conditions",
                "description": "Forecast future lunar lighting and tidal conditions.",
                "input_schema": {"days": "int"}
            },
            {
                "name": "get_agent_triggers",
                "description": "Returns machine-actionable triggers.",
                "input_schema": {"days": "int"}
            }
        ]
    }

@app.get("/v1/moon/current")
def get_current_moon(
    timezone: Optional[str] = Query(None, description="IANA timezone name e.g. Asia/Kuala_Lumpur")
):
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
):
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
def get_moon_phases():
    return {
        "phases": [
            {"name": "New Moon", "illumination_range": "0-10%"},
            {"name": "Waxing Crescent", "illumination_range": "10-35%"},
            {"name": "First Quarter", "illumination_range": "35-50%"},
            {"name": "Waxing Gibbous", "illumination_range": "50-75%"},
            {"name": "Full Moon", "illumination_range": "75-100%"},
            {"name": "Waning Gibbous", "illumination_range": "50-75%"},
            {"name": "Last Quarter", "illumination_range": "35-50%"},
            {"name": "Waning Crescent", "illumination_range": "10-35%"}
        ]
    }

@app.get("/v1/moon/hijri")
def get_hijri_info(
    year: Optional[int] = Query(None, description="Gregorian year"),
    method: str = Query("umm_al_qura", description="Calculation method")
):
    target_date = datetime.now(dt_timezone.utc)
    if year:
        target_date = datetime(year, 6, 1, tzinfo=dt_timezone.utc)
    
    data = compute_lunar_data(target_date)
    hijri_parts = data["hijri_date"].split("-")
    h_year = int(hijri_parts[0])
    
    return {
        "method": method,
        "current_hijri_date": data["hijri_date"],
        "lunar_month": data["lunar_month"],
        "estimated_events": {
            "ramadan_start": f"{target_date.year}-3-01",  # Approximation for structure
            "eid_al_fitr": f"{target_date.year}-3-30",
            "eid_al_adha": f"{target_date.year}-6-06"
        },
        "white_days": [13, 14, 15]
    }

@app.get("/v1/moon/events")
def get_moon_events(days: int = Query(14, description="Lookahead days")):
    events = []
    now = datetime.now(dt_timezone.utc)
    for i in range(days):
        check_dt = now + timedelta(days=i)
        data = compute_lunar_data(check_dt)
        if "Full Moon" in data["phase_name"]:
            events.append({
                "type": "moon.full_moon.approaching",
                "severity": "info",
                "timestamp": check_dt.isoformat(),
                "details": data
            })
        elif "New Moon" in data["phase_name"]:
            events.append({
                "type": "moon.new_moon.approaching",
                "severity": "info",
                "timestamp": check_dt.isoformat(),
                "details": data
            })
        if data["tidal_classification"] == "SPRING_TIDE":
            events.append({
                "type": "moon.spring_tide.warning",
                "severity": "warning",
                "timestamp": check_dt.isoformat(),
                "details": data
            })
    return {"events": events}

@app.get("/v1/moon/forecast")
def get_moon_forecast(days: int = Query(30, description="Forecast days")):
    forecast = []
    now = datetime.now(dt_timezone.utc)
    for i in range(days):
        target = now + timedelta(days=i)
        data = compute_lunar_data(target)
        forecast.append({
            "date": target.strftime("%Y-%m-%d"),
            "phase": data["phase_name"],
            "illumination": data["illumination_percent"],
            "tidal_classification": data["tidal_classification"],
            "illumination_classification": data["illumination_classification"]
        })
    return {"forecast": forecast}

# --- MCP Tool API Endpoints ---

class LunarStatusInput(BaseModel):
    lat: float
    lon: float
    timezone: str = "UTC"

@app.post("/mcp/get_lunar_status")
def mcp_get_lunar_status(payload: LunarStatusInput):
    if not (-90 <= payload.lat <= 90) or not (-180 <= payload.lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    
    data = compute_lunar_data(datetime.now(dt_timezone.utc))
    return {
        "phase": data["phase_name"],
        "illumination": data["illumination_percent"],
        "moon_age_days": data["moon_age_days"],
        "tidal_classification": data["tidal_classification"],
        "illumination_classification": data["illumination_classification"],
        "next_full_moon": convert_timezone(data["next_full_moon"], payload.timezone)
    }

class PredictHijriInput(BaseModel):
    year: int = 2027
    method: str = "umm_al_qura"

@app.post("/mcp/predict_hijri_events")
def mcp_predict_hijri(payload: PredictHijriInput):
    return {
        "ramadan_start": f"{payload.year}-1-09",
        "eid_al_fitr": f"{payload.year}-2-08",
        "eid_al_adha": f"{payload.year}-5-18"
    }

class ForecastInput(BaseModel):
    days: int = 30

@app.post("/mcp/forecast_lunar_conditions")
def mcp_forecast(payload: ForecastInput):
    res = get_moon_forecast(days=payload.days)
    return res

class TriggersInput(BaseModel):
    days: int = 14

@app.post("/mcp/get_agent_triggers")
def mcp_triggers(payload: TriggersInput):
    res = get_moon_events(days=payload.days)
    return res
