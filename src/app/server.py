"""Step 8: FastAPI app - DCRNN sensor forecasts, congestion-coloured routes, Folium map."""
import json
import logging
import pickle
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.app.map_builder import build_base_map, node_color, node_var_names
from src.config import BBOX, GRAPH_DIR, PROJECT_ROOT
from src.inference.forecast_io import FORECAST_DIR, save_forecast
from src.inference.loader import load_model
from src.inference.predict import STALE_AFTER_MIN, predict_once
from src.routing.osm_loader import get_osm_graph
from src.routing.pathfinder import find_candidate_paths
from src.routing.spreading import colorize_path, path_score
from src.tomtom.buffer import RollingBuffer
from src.tomtom.poller import BUFFER_PATH, load_reference_free_flow

log = logging.getLogger("app")
TEMPLATES = Jinja2Templates(directory=str(PROJECT_ROOT / "src" / "app" / "templates"))
GOOGLE_MAPS_URL = "https://www.google.com/maps/@12.9716,77.5946,12z/data=!5m1!1e1"
VALID_HORIZONS = (5, 10, 20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.config = load_model()          # once, not per request
    app.state.free_flows = load_reference_free_flow()
    app.state.lock = threading.Lock()                          # one inference at a time
    app.state.nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv").sort_values("node_id")
    app.state.osm = get_osm_graph()                            # cached; first ever build is slow
    base_map = build_base_map(app.state.nodes_df)
    root = base_map.get_root()
    root.render()
    app.state.map_parts = {"header": root.header.render(), "html": root.html.render(),
                           "script": root.script.render(), "map_var": base_map.get_name(),
                           "node_vars": node_var_names(base_map)}
    if not BUFFER_PATH.exists():
        log.warning("no buffer at %s yet - start the poller (step 6) for live forecasts",
                    BUFFER_PATH)
    yield


app = FastAPI(title="Bangalore DCRNN traffic", lifespan=lifespan)


# ---------------------------------------------------------------------- forecast selection
def _load_buffer() -> RollingBuffer | None:
    try:
        return RollingBuffer.load(BUFFER_PATH)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError):
        return None                       # missing, or caught mid-write by the poller


def _latest_saved() -> dict | None:
    for path in sorted(FORECAST_DIR.glob("forecast_*.json"), reverse=True):
        try:
            return json.loads(path.read_text())
        except ValueError:
            continue
    return None


def _staleness_warning(input_last_poll_at: str, now: datetime) -> str | None:
    age = (now - datetime.fromisoformat(input_last_poll_at)).total_seconds() / 60
    return (f"newest poll is {age:.0f} min old; forecast is stale - is the poller running?"
            if age > STALE_AFTER_MIN else None)


def get_forecast(app: FastAPI) -> dict:
    """Reuse the saved forecast when it was made from the buffer's newest poll, else predict."""
    now = datetime.now()
    saved, buffer = _latest_saved(), _load_buffer()
    ready = buffer is not None and buffer.is_ready()

    if ready:
        newest = buffer.snapshot()[1][-1].isoformat()
        if saved is not None and saved["input_last_poll_at"] == newest:
            result = saved                                   # same inputs -> same forecast
        else:
            with app.state.lock:
                result = predict_once(app.state.model, buffer, app.state.free_flows)
            save_forecast(result, FORECAST_DIR)
    elif saved is not None:
        result = saved
        result["warnings"] = [*result.get("warnings", []),
                              "the live buffer is not ready; showing the last saved forecast"]
    else:
        raise HTTPException(503, "No forecast available: run the Step 6 warmup or poller first.")

    result = dict(result)
    warnings = [w for w in result.get("warnings", []) if not w.startswith("newest poll is")]
    stale = _staleness_warning(result["input_last_poll_at"], now)
    result["warnings"] = warnings + ([stale] if stale else [])
    return result


def shape_for_horizon(result: dict, horizon: int) -> dict:
    label = f"plus_{horizon}min"
    nodes = []
    for n in result["nodes"]:
        pred, ff = n[f"pred_{label}_kmh"], n["free_flow_kmh"]
        band = n[f"band_{label}"]
        nodes.append({"node_id": n["node_id"], "name": n["name"], "lat": n["lat"],
                      "lon": n["lon"], "current_kmh": n["current_kmh"], "pred_kmh": pred,
                      "free_flow_kmh": ff,
                      "congestion_pct": round(float(np.clip((1 - pred / ff) * 100, 0, 100)), 1),
                      "band": band, "color": node_color(band)})
    return {"forecast_made_at": result["forecast_made_at"],
            "input_last_poll_at": result["input_last_poll_at"], "horizon_min": horizon,
            "warnings": result.get("warnings", []), "nodes": nodes}


# ---------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(request, "index.html", {
        **request.app.state.map_parts, "google_maps_url": GOOGLE_MAPS_URL})


@app.get("/api/predict")
@app.get("/api/forecast")                      # kept from step 8-lite
def api_predict(horizon: int = Query(5)):
    if horizon not in VALID_HORIZONS:
        raise HTTPException(400, f"horizon must be one of {list(VALID_HORIZONS)}")
    return shape_for_horizon(get_forecast(app), horizon)


class RouteRequest(BaseModel):
    src_lat: float
    src_lon: float
    dst_lat: float
    dst_lon: float
    horizon_min: int = 5
    k_paths: int = Field(3, ge=1, le=3)


def _inside_bbox(lat: float, lon: float) -> bool:
    return BBOX["south"] <= lat <= BBOX["north"] and BBOX["west"] <= lon <= BBOX["east"]


@app.post("/api/route")
def api_route(req: RouteRequest):
    if req.horizon_min not in VALID_HORIZONS:
        raise HTTPException(400, f"horizon_min must be one of {list(VALID_HORIZONS)}")
    for label, lat, lon in (("source", req.src_lat, req.src_lon),
                            ("destination", req.dst_lat, req.dst_lon)):
        if not _inside_bbox(lat, lon):
            raise HTTPException(400, f"{label} is outside the covered area "
                                     f"(lat {BBOX['south']}-{BBOX['north']}, "
                                     f"lon {BBOX['west']}-{BBOX['east']})")
    forecast = get_forecast(app)                                   # 503 if none available
    predictions = shape_for_horizon(forecast, req.horizon_min)
    paths = find_candidate_paths(app.state.osm, (req.src_lat, req.src_lon),
                                 (req.dst_lat, req.dst_lon), req.k_paths)
    if not paths:
        raise HTTPException(404, "No drivable route found between those points")

    scored = []
    for path in paths:
        segments = colorize_path(path, predictions, app.state.nodes_df)
        scored.append({**path_score(segments), "segments": segments})
    scored.sort(key=lambda p: p["total_time_min"])
    for rank, p in enumerate(scored, start=1):
        p["rank"] = rank
    return {"forecast_made_at": forecast["forecast_made_at"],
            "input_last_poll_at": forecast["input_last_poll_at"],
            "horizon_min": req.horizon_min, "warnings": forecast.get("warnings", []),
            "paths": scored}


PRESETS = [
    ("Marathahalli -> Silk Board (ORR lunch)", "Marathahalli Bridge", "Silk Board"),
    ("Hebbal -> M.G. Road (morning rush)", "Hebbal Flyover", "M.G. Road Trinity"),
    ("Indiranagar -> Jayanagar (CBD)", "Indiranagar 100Ft", "Jayanagar 4th Block"),
    ("Whitefield -> Majestic (east-west)", "Whitefield ITPL", "Majestic"),
    ("Silk Board -> Marathahalli (evening ORR)", "Silk Board", "Marathahalli Bridge"),
]


@app.get("/api/presets")
def api_presets():
    from src.nodes import NODES

    by_name = {n.name: n for n in NODES}
    point = lambda name: {"lat": by_name[name].lat, "lon": by_name[name].lon}
    return [{"label": label, "src": point(a), "dst": point(b)} for label, a, b in PRESETS]


@app.get("/api/health")
def api_health():
    buffer = _load_buffer()
    saved = _latest_saved()
    return {"status": "ok", "buffer_ready": bool(buffer is not None and buffer.is_ready()),
            "latest_forecast_at": saved["forecast_made_at"] if saved else None}
