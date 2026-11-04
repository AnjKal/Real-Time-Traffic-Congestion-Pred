"""Web app + routing API (modern Python). Talks to the model service over HTTP.

    python run_webapp.py            # then open http://127.0.0.1:8010/

Endpoints
    GET  /                          the page
    GET  /api/status                model-service state (mode, readiness, horizons)
    GET  /api/forecast?horizon_min= forecast for all junctions (proxied from the model service)
    POST /api/route                 {src_lat, src_lon, dst_lat, dst_lon, horizon_min, k_paths}
    GET  /api/presets               one-click source/destination pairs
"""
import os

import folium
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from webapp import routing
from webapp.osm_graph import PROJECT_ROOT, SPOTS_PATH, get_graph

MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://127.0.0.1:8001")
TEMPLATES = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "webapp", "templates"))
GOOGLE_MAPS_URL = "https://www.google.com/maps/@12.9716,77.5946,12z/data=!5m1!1e1"
BAND_COLORS = {"free": "#2ecc71", "moderate": "#f1c40f", "heavy": "#e74c3c"}
# (label, source junction row, destination junction row) in bangalore_traffic_spots.csv
PRESET_ROWS = [
    ("Marathahalli -> Silk Board (ORR)", 7, 0),
    ("Hebbal -> MG Road", 25, 17),
    ("Indiranagar -> Jayanagar", 14, 46),
    ("Whitefield (ITPL) -> Majestic", 9, 22),
    ("Silk Board -> Marathahalli (ORR)", 0, 7),
]


class RouteRequest(BaseModel):
    src_lat: float
    src_lon: float
    dst_lat: float
    dst_lon: float
    horizon_min: int
    k_paths: int = Field(3, ge=1, le=3)


def create_app(model_service_url=None, graph=None, spots=None):
    """graph / spots injectable for tests."""
    url = (model_service_url or MODEL_SERVICE_URL).rstrip("/")
    state = {"graph": graph, "spots": spots}
    app = FastAPI(title="Bangalore congestion forecast")

    def spots_df():
        if state["spots"] is None:
            state["spots"] = pd.read_csv(SPOTS_PATH)
        return state["spots"]

    def road_graph():
        if state["graph"] is None:
            state["graph"] = get_graph()
        return state["graph"]

    def call_model(path, params=None):
        try:
            r = requests.get(url + path, params=params, timeout=20)
        except requests.RequestException:
            raise HTTPException(502, "Cannot reach the model service at %s. Start it with: "
                                     "python model_service.py" % url)
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise HTTPException(r.status_code, detail)
        return r.json()

    def base_map():
        s = spots_df()
        m = folium.Map(location=[s.lat.mean(), s.lon.mean()], zoom_start=11,
                       tiles="OpenStreetMap", width="100%", height="100%")
        names = {}
        for i, row in s.iterrows():
            marker = folium.CircleMarker(
                location=(row.lat, row.lon), radius=7, color="#333333", weight=1, fill=True,
                fill_color="#95a5a6", fill_opacity=0.8, tooltip="%s - waiting for forecast" % row["name"])
            marker.add_to(m)
            names[i] = marker.get_name()
        root = m.get_root()
        root.render()
        return {"header": root.header.render(), "html": root.html.render(),
                "script": root.script.render(), "map_var": m.get_name(), "junction_vars": names}

    parts = {}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        if not parts:
            parts.update(base_map())
        return TEMPLATES.TemplateResponse(request, "index.html", {
            **parts, "google_maps_url": GOOGLE_MAPS_URL, "band_colors": BAND_COLORS})

    @app.get("/api/status")
    def status():
        return call_model("/health")

    @app.get("/api/forecast")
    def forecast(horizon_min: int = None):
        return call_model("/forecast", None if horizon_min is None else {"horizon_min": horizon_min})

    @app.get("/api/presets")
    def presets():
        s = spots_df()
        pt = lambda i: {"lat": float(s.iloc[i].lat), "lon": float(s.iloc[i].lon),
                        "name": str(s.iloc[i]["name"])}
        return [{"label": label, "src": pt(a), "dst": pt(b)} for label, a, b in PRESET_ROWS]

    @app.post("/api/route")
    def route(req: RouteRequest):
        forecast_json = call_model("/forecast", {"horizon_min": req.horizon_min})
        junctions = forecast_json["nodes"]
        try:
            candidate_paths = routing.find_candidate_paths(
                road_graph(), (req.src_lat, req.src_lon), (req.dst_lat, req.dst_lon), req.k_paths)
        except routing.RoutingError as e:
            raise HTTPException(400, str(e))
        scored = []
        for edges in candidate_paths:
            segments = routing.colorize_path(edges, junctions)
            scored.append({**routing.path_score(segments), "segments": segments})
        scored.sort(key=lambda p: p["total_time_min"])
        for rank, p in enumerate(scored, start=1):
            p["rank"] = rank
        return {"horizon_min": req.horizon_min, "mode": forecast_json.get("mode"),
                "warning": forecast_json.get("warning"),
                "input_last_slot": forecast_json["input_last_slot"],
                "forecast_for": forecast_json["forecast_for"],
                "data_age_min": forecast_json.get("data_age_min"), "paths": scored}

    return app


app = create_app()
