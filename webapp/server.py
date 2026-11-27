"""Web app + routing API (modern Python). Talks to the model service over HTTP.

    python run_webapp.py            # then open http://127.0.0.1:8010/

Endpoints
    GET  /                          the page
    GET  /api/status                model-service state (mode, readiness, horizons)
    GET  /api/forecast?horizon_min= forecast for all junctions (proxied from the model service)
    POST /api/route                 {src_lat, src_lon, dst_lat, dst_lon, horizon_min, k_paths}
    GET  /api/presets               one-click source/destination pairs
    GET  /api/features              area-wise aggregated traffic (from traffic.features)
    GET  /api/incidents             latest congestion-event snapshot (from the incident agent)
    POST /api/journey               start watching a chosen route (journey intelligence agent)
    GET  /api/journey/{id}          the agent's latest decision for that journey
    POST /api/journey/{id}/feedback {"response": "accepted" | "dismissed"}
    POST /api/journey/{id}/switch   adopt the alternative the agent recommended
    POST /api/journey/{id}/position {"lat": .., "lon": ..} optional real position feed
    DELETE /api/journey/{id}        stop watching
    GET  /api/journeys              every watched journey (agent state, for debugging)

    # interaction agent (Phase 4) -- see webapp/interaction_api.py
    POST   /api/interaction/session        start an empty conversation; issues a session token
    POST   /api/chat                       ask the interaction agent; issues a session token
    GET    /api/chat/history               this conversation's messages (Bearer token required)
    DELETE /api/chat/history               forget this conversation's messages
    GET    /api/notifications              this conversation's proactive alerts
    GET    /api/notifications/unread-count unread alert count
    POST   /api/notifications/read         mark one (or all) alerts read
    GET    /api/subscriptions              journeys this conversation is subscribed to
    POST   /api/subscriptions              subscribe explicitly to a watched journey
    DELETE /api/subscriptions/{journey_id} stop monitoring a journey
    GET    /api/interaction/status         interaction agent + notification runtime health
"""
import json
import logging
import os
from contextlib import asynccontextmanager

import folium
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from webapp import live_readings, routing
from webapp.interaction_api import build_interaction_services, create_interaction_router
from webapp.journey_agent import JourneyAgent
from webapp.journey_runtime import JourneyRuntime
from webapp.osm_graph import PROJECT_ROOT, SPOTS_PATH, get_graph

LOG = logging.getLogger(__name__)

# Latest area-wise features, written by the feature aggregator to the shared data volume.
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "live", "features.json")
INCIDENTS_PATH = os.path.join(PROJECT_ROOT, "data", "live", "incidents.json")

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


class JourneyRequest(RouteRequest):
    """A route the traveller actually chose; rank is the one they picked on the page."""
    rank: int = Field(1, ge=1, le=3)
    label: str = None
    # Start journey puts a simulated car on the route. minutes_per_second compresses time:
    # 0.5 is one simulated minute per two real seconds. Set simulate false to watch a route
    # without a car, in which case progress is estimated from elapsed time instead.
    simulate: bool = True
    minutes_per_second: float = Field(None, gt=0, le=10)


class FeedbackRequest(BaseModel):
    response: str = Field(..., pattern="^(accepted|dismissed)$")


class PositionRequest(BaseModel):
    lat: float
    lon: float


def create_app(model_service_url=None, graph=None, spots=None, journey_agent=None,
               interaction=None):
    """graph / spots / journey_agent injectable for tests.

    The journey agent runs in this process because deciding whether a better route exists needs
    the road graph, which only the web app has. Its Kafka consumers/producer are started by
    JourneyRuntime from the lifespan below, on consumer groups of their own.

    The interaction agent (Phase 4) is wired here too: ``interaction`` is an
    :class:`~webapp.interaction_api.InteractionServices` built from ``INTERACTION_*`` settings, and
    may be injected by tests. Building it makes no AWS call and starts nothing; its notification
    runtime is started/stopped by the lifespan, and a failure to build it must not stop the app --
    its endpoints simply report unavailable.
    """
    url = (model_service_url or MODEL_SERVICE_URL).rstrip("/")
    state = {"graph": graph, "spots": spots}
    agent = journey_agent or JourneyAgent(graph_provider=lambda: road_graph())
    runtime = JourneyRuntime(agent, model_service_url=url)

    interaction_services = interaction
    if interaction_services is None:
        try:
            interaction_services = build_interaction_services()
        except Exception:                                         # noqa: BLE001 - never fatal
            LOG.exception("Could not build the interaction agent; its endpoints will report "
                          "unavailable and the rest of the app continues")
            interaction_services = None

    @asynccontextmanager
    async def lifespan(_app):
        runtime.start()
        if interaction_services is not None:
            interaction_services.start()
        try:
            yield
        finally:
            if interaction_services is not None:
                interaction_services.stop()
            runtime.stop()

    app = FastAPI(title="Bangalore congestion forecast", lifespan=lifespan)
    app.state.journey_agent = agent
    app.state.journey_runtime = runtime
    app.state.interaction = interaction_services
    app.include_router(create_interaction_router(interaction_services))

    def spots_df():
        if state["spots"] is None:
            state["spots"] = pd.read_csv(SPOTS_PATH)
        return state["spots"]

    def road_graph():
        if state["graph"] is None:
            try:
                state["graph"] = get_graph()
            except FileNotFoundError:
                raise HTTPException(503, "The road map has not been built yet. "
                                         "Run this once: python -m webapp.osm_graph")
        return state["graph"]

    def call_model(path, params=None):
        try:
            r = requests.get(url + path, params=params, timeout=20)
        except requests.RequestException:
            raise HTTPException(502, "Cannot reach the model service at %s. Start it with: "
                                     "docker compose up -d" % url)
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

    def build_routes(req):
        """Candidate routes scored for one request. Shared by /api/route and /api/journey so a
        journey watches exactly the route the page displayed."""
        forecast_json = call_model("/forecast", {"horizon_min": req.horizon_min})
        junctions = forecast_json["nodes"]
        try:
            candidate_paths = routing.find_candidate_paths(
                road_graph(), (req.src_lat, req.src_lon), (req.dst_lat, req.dst_lon), req.k_paths)
        except routing.RoutingError as e:
            raise HTTPException(400, str(e))
        # Latest TomTom free-flow speed per junction (read-only from the readings archive). Never
        # substituted by reference_kmh; absent values just leave free_flow_speed_kmh/delay null.
        free_flow, free_flow_as_of = live_readings.latest_free_flow_by_junction()
        scored = []
        for edges in candidate_paths:
            segments = routing.colorize_path(edges, junctions, free_flow_by_junction=free_flow)
            scored.append({**routing.path_score(segments), "segments": segments})
        scored.sort(key=lambda p: p["total_time_min"])
        for rank, p in enumerate(scored, start=1):
            p["rank"] = rank
        return forecast_json, scored, free_flow_as_of

    @app.post("/api/route")
    def route(req: RouteRequest):
        forecast_json, scored, free_flow_as_of = build_routes(req)
        return {"horizon_min": req.horizon_min, "mode": forecast_json.get("mode"),
                "warning": forecast_json.get("warning"),
                "warmup_steps": forecast_json.get("warmup_steps"),
                "input_window_steps": forecast_json.get("input_window_steps"),
                "warmup_warning": forecast_json.get("warmup_warning"),
                "sensor_coverage_pct": forecast_json.get("sensor_coverage_pct"),
                "input_last_slot": forecast_json["input_last_slot"],
                "forecast_for": forecast_json["forecast_for"],
                "free_flow_as_of": free_flow_as_of,
                "data_age_min": forecast_json.get("data_age_min"), "paths": scored}

    @app.get("/api/features")
    def features():
        """Latest area-wise aggregated traffic from the traffic.features topic (written by the
        feature aggregator to data/live/features.json). 503 until the first batch is produced."""
        try:
            with open(FEATURES_PATH, "r") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            raise HTTPException(503, "No area features yet. The feature aggregator publishes to "
                                     "traffic.features after the first consumed poll.")
        except (ValueError, OSError):
            raise HTTPException(503, "Area features are being written; try again shortly.")
        return payload

    @app.get("/api/incidents")
    def incidents():
        """Latest congestion-event snapshot, written by the Kafka incident agent."""
        try:
            with open(INCIDENTS_PATH, "r") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            raise HTTPException(503, "No congestion events yet. The incident agent publishes "
                                     "after it joins the first raw poll and forecast.")
        except (ValueError, OSError):
            raise HTTPException(503, "Congestion events are being written; try again shortly.")
        return payload

    # ----------------------------------------------------------------- journey agent
    def _journey_summary(journey):
        """The journey without its bulky geometry; the page already has the drawn route."""
        if journey is None:
            return None
        skip = ("route_polyline", "route_spine", "spine", "route_nodes", "route_node_cum_m",
                "pending_alternative")
        return {key: value for key, value in journey.items() if key not in skip}

    @app.post("/api/journey")
    def start_journey(req: JourneyRequest):
        """Start watching the route the traveller chose, and return the agent's first decision.

        The route is rebuilt here with the same code the map uses, so the agent watches exactly
        what the page displayed rather than a re-derived approximation.
        """
        _, scored, _ = build_routes(req)
        chosen = next((path for path in scored if path["rank"] == req.rank), None)
        if chosen is None:
            raise HTTPException(400, "route rank %d was not among the %d routes found"
                                     % (req.rank, len(scored)))
        score = {key: value for key, value in chosen.items() if key != "segments"}
        journey = agent.register({"lat": req.src_lat, "lon": req.src_lon},
                                 {"lat": req.dst_lat, "lon": req.dst_lon},
                                 horizon_min=req.horizon_min, segments=chosen["segments"],
                                 score=score, rank=req.rank, label=req.label)
        if req.simulate:
            started = runtime.simulator.start_journey(
                journey["journey_id"], minutes_per_second=req.minutes_per_second)
            if started is not None:
                journey = started
        return {"journey": _journey_summary(journey),
                "decision": agent.latest(journey["journey_id"])}

    @app.get("/api/journey/{journey_id}")
    def journey_state(journey_id: str):
        """The agent's latest decision for one journey. 404 once it has been forgotten, which
        tells the page to drop its stored journey."""
        journey = agent.memory.recall(journey_id)
        decision = agent.latest(journey_id)
        if journey is None and decision is None:
            raise HTTPException(404, "no journey %s is being watched" % journey_id)
        return {"journey": _journey_summary(journey), "decision": decision,
                "history": agent.memory.decisions(journey_id, limit=5)}

    @app.get("/api/journey/{journey_id}/route")
    def journey_route(journey_id: str):
        """The journey's own route as a line of points, so the page can draw what is being
        driven. After a switch this is the adopted road, which is no longer one of the
        candidates /api/route returns for the original source."""
        journey = agent.memory.recall(journey_id)
        if journey is None:
            raise HTTPException(404, "no journey %s is being watched" % journey_id)
        return {"journey_id": journey_id,
                "polyline": [[point[0], point[1]] for point in
                             (journey.get("route_polyline") or [])],
                "total_length_km": journey.get("total_length_km"),
                "switch_count": journey.get("switch_count", 0)}

    @app.post("/api/journey/{journey_id}/feedback")
    def journey_feedback(journey_id: str, req: FeedbackRequest):
        """Record how the traveller answered. A dismissal raises the agent's own thresholds and
        starts a cooldown, so repeating the same advice is suppressed."""
        preferences = agent.feedback(journey_id, req.response)
        if preferences is None:
            raise HTTPException(404, "no journey %s is being watched" % journey_id)
        return {"journey_id": journey_id, "response": req.response, "preferences": preferences}

    @app.post("/api/journey/{journey_id}/switch")
    def journey_switch(journey_id: str):
        """Adopt the alternative the agent recommended; its score becomes the new baseline."""
        decision = agent.latest(journey_id)
        alternative = (decision or {}).get("alternative")
        updated = agent.adopt_alternative(journey_id)
        if updated is None:
            raise HTTPException(400, "there is no alternative route to switch to right now")
        return {"journey": _journey_summary(updated),
                "segments": (alternative or {}).get("segments") or [],
                "score": (alternative or {}).get("score") or {},
                "decision": agent.latest(journey_id)}

    @app.post("/api/journey/{journey_id}/position")
    def journey_position(journey_id: str, req: PositionRequest):
        """Optional real position feed. Without it the agent estimates progress from elapsed
        time; with it, the estimate is replaced and nothing else in the policy changes."""
        journey = agent.memory.recall(journey_id)
        if journey is None:
            raise HTTPException(404, "no journey %s is being watched" % journey_id)
        cum_m = routing.cum_at_position(journey.get("route_polyline") or [], req.lat, req.lon)
        if cum_m is None:
            raise HTTPException(400, "this journey has no stored route geometry")
        agent.memory.update_journey(journey_id, position_cum_m=cum_m)
        return {"journey_id": journey_id, "position_cum_m": cum_m}

    @app.delete("/api/journey/{journey_id}")
    def end_journey(journey_id: str):
        return {"journey_id": journey_id, "ended": bool(agent.end(journey_id))}

    @app.get("/api/journeys")
    def journeys():
        """Every watched journey and its latest decision (the agent's own state)."""
        return agent.snapshot()

    @app.get("/api/switches")
    def switches(journey_id: str = None):
        """The switch ledger: every reroute taken, what it claimed to save, and what it saved.

        Rows outlive the journey on purpose. ``gain_min`` is what the agent promised at the
        moment of the switch; ``gain_realised_min`` is filled in on arrival, so the two columns
        together say whether the advice was worth taking.
        """
        rows = agent.memory.switches(journey_id)
        claimed = sum(float(row.get("gain_min") or 0.0) for row in rows)
        realised = [row for row in rows if row.get("gain_realised_min") is not None]
        return {"switch_count": len(rows),
                "claimed_gain_min": round(claimed, 1),
                "scored_count": len(realised),
                "realised_gain_min": round(sum(float(row["gain_realised_min"])
                                               for row in realised), 1),
                "switches": rows}

    @app.get("/api/journey-runtime")
    def journey_runtime():
        """How the agent is being fed right now (Kafka or polling), for debugging."""
        return runtime.status()

    return app


app = create_app()
