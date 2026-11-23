"""The journey intelligence agent: is the route this traveller chose still the right one?

No agent framework. A plain dictionary of tools (``TOOLS`` below), a loop that calls them in a
fixed order, and a record of every call that is published with the decision and shown on the
page. Nothing here asks a language model anything; every threshold is a number in this file.

What happens each time a fresh forecast is available, for one watched journey:

  0 observe   recall_memory, estimate_progress, get_free_flow
  1 assess    score_current_route (the part still ahead), check_incidents_on_route
  2 explore   find_alternatives -- only once a trigger has fired, because this is the one
              expensive step: it re-runs the OpenStreetMap search from the car's position
  3 decide    hold / watch / notify_delay / recommend_switch / arrived / expired
  4 reflect   remember_decision

The comparison that matters is on the clock, not on minutes remaining. Minutes remaining always
falls as you drive, so a journey can rot from 32 to 48 minutes without any single check seeing a
rise. So the agent freezes one number when the traveller chooses a route -- the arrival time it
promised them -- and every later check asks only: are you now due to arrive later than that?
"""
from __future__ import absolute_import

import hashlib
import logging
import time
import uuid
from datetime import timedelta

from webapp import routing
from webapp.journey_memory import (JourneyMemory, iso, parse_iso, slim_decision, utc_now)

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- operating rules
# Transparent thresholds, not calibrated probabilities. Every one of them exists to stop the
# agent saying something a real driver would find useless or unsafe.

MIN_GAIN_MIN = 3.0            # an alternative must save at least this many minutes ...
MIN_GAIN_SHARE = 0.10         # ... or this share of the journey still left, whichever is larger
INCIDENT_GAIN_MIN = 2.0       # a reported jam on the road ahead lowers the bar to this
TRIGGER_LATE_MIN = 3.0        # how late you must be before it bothers searching for alternatives
TRIGGER_HEAVY_POINTS = 15.0   # ... or this many more percentage points of the route gone heavy
NOTIFY_LATE_MIN = 5.0         # tell them they are late even when no better route exists
WARNING_DISTANCE_M = 250.0    # the turn must be at least this far ahead ...
WARNING_TIME_MIN = 0.5        # ... and this long ahead, so there is time to react
# Those two started at 500 m and a full minute. A simulated run showed the cost: a bypass saving
# 19 minutes sat 460 m ahead at 42 km/h -- 39 seconds of notice, which is more warning than a
# satnav usually gives -- and the rule threw it away, after which the car crawled for another 42
# minutes with no alternative left to offer. A floor that rejects usable advice is worse than no
# floor, so it is set where a real instruction would be given.
ENDGAME_ETA_MIN = 6.0         # no rerouting once this close to the destination, by time ...
ENDGAME_DISTANCE_M = 1500.0   # ... or by distance
SWITCH_STREAK = 2             # the same alternative must win this many checks in a row
ALERT_COOLDOWN_TICKS = 3      # minimum checks between interruptions
MAX_SWITCH_SUGGESTIONS = 3    # per journey; a nav that nags gets switched off
MIN_COVERAGE_PCT = 70.0       # below this much live sensor data on the route, stay quiet
SAME_ROUTE_OVERLAP = 0.80     # a candidate sharing more than this with your route IS your route
BACKTRACK_WINDOW_M = 300.0    # a candidate re-using road already driven inside this is a U-turn
CANDIDATE_PATHS = 5           # ask the road graph for more than the three the map shows
ARRIVAL_REMAINING_M = 60.0    # this close to the end counts as arrived
ARRIVAL_GRACE_MIN = 30.0      # stop watching this long after the expected arrival
DISMISSAL_FACTOR = 1.5        # each "Keep my route" multiplies the thresholds by this ...
MAX_DISMISSAL_SCALE = 4.0     # ... up to this ceiling

TOOLS = ("recall_memory", "estimate_progress", "get_free_flow", "score_current_route",
         "check_incidents_on_route", "find_alternatives", "remember_decision")

ACTIVE_INCIDENT_STATES = ("active", "recovering")
WATCH_INCIDENT_STATES = ("watch",)


def _round(value, digits=1):
    return None if value is None else round(float(value), digits)


def _signature(nodes):
    """Short stable id for a candidate route, so "the same alternative as last time" is cheap."""
    raw = ",".join(str(node) for node in nodes)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class ForecastBook(object):
    """One poll's forecasts, all horizons, indexed for lookup by junction and by how far ahead.

    This is what lets a simulated car travel through time. A single poll already carries twelve
    forecasts -- in the docker test cadence, +2 through +24 minutes -- so "what will this road be
    like ten minutes from now" is a number already in hand, with nothing to wait for.
    """

    def __init__(self, poll_id, by_horizon, meta=None):
        self.poll_id = str(poll_id)
        self.horizons = sorted(int(h) for h in by_horizon)
        self._by_horizon = {int(h): value for h, value in by_horizon.items()}
        self.meta = meta or {}

    @classmethod
    def from_prediction_message(cls, message):
        """Build from a traffic.predictions envelope (record_type 'forecast')."""
        forecasts = message.get("forecasts") or {}
        if not forecasts:
            return None
        by_horizon = {}
        for horizon, payload in forecasts.items():
            nodes = [node for node in payload.get("nodes", []) if node.get("junction_id")]
            if nodes:
                by_horizon[int(horizon)] = nodes
        if not by_horizon:
            return None
        first = forecasts[str(min(int(h) for h in forecasts))]
        meta = {key: first.get(key) for key in
                ("made_at", "step_minutes", "mode", "input_last_slot", "forecast_for",
                 "sensor_coverage_pct", "warmup_steps", "data_age_min", "warning",
                 "warmup_warning")}
        return cls(message.get("poll_id") or message.get("made_at") or "", by_horizon, meta)

    @classmethod
    def from_single_forecast(cls, payload, poll_id=None):
        """Build from one /forecast response. Only one horizon, so no time travel."""
        nodes = [node for node in payload.get("nodes", []) if node.get("junction_id")]
        if not nodes:
            return None
        horizon = int(payload.get("horizon_min") or payload.get("horizons_min", [0])[0] or 0)
        meta = {key: payload.get(key) for key in
                ("made_at", "step_minutes", "mode", "input_last_slot", "forecast_for",
                 "sensor_coverage_pct", "warmup_steps", "data_age_min", "warning",
                 "warmup_warning")}
        return cls(poll_id or payload.get("input_last_slot") or "", {horizon: nodes}, meta)

    def merge_horizon(self, horizon, payload):
        """Add another horizon fetched separately over HTTP."""
        nodes = [node for node in payload.get("nodes", []) if node.get("junction_id")]
        if nodes:
            self._by_horizon[int(horizon)] = nodes
            self.horizons = sorted(self._by_horizon)

    def horizon_for(self, offset_min):
        """The available horizon closest to ``offset_min`` minutes from now."""
        target = max(0.0, float(offset_min or 0.0))
        return min(self.horizons, key=lambda h: (abs(h - target), h))

    def node_list(self, offset_min=0.0):
        return self._by_horizon[self.horizon_for(offset_min)]

    def node_table(self, offset_min=0.0):
        return {node["junction_id"]: node for node in self.node_list(offset_min)}

    def lookup(self, base_offset_min=0.0):
        """A ``(junction_id, offset_min)`` callable for routing.score_spine.

        ``base_offset_min`` is how far the journey clock has already run past this forecast;
        ``offset_min`` is how long the car still has to drive before reaching that junction. The
        sum is when the car is actually there, which is the forecast it should be priced at.
        """
        cache = {}

        def pick(junction_id, offset_min):
            horizon = self.horizon_for(float(base_offset_min) + float(offset_min or 0.0))
            table = cache.get(horizon)
            if table is None:
                table = {node["junction_id"]: node for node in self._by_horizon[horizon]}
                cache[horizon] = table
            return table.get(junction_id)

        return pick

    @property
    def sensor_coverage_pct(self):
        return self.meta.get("sensor_coverage_pct")

    @property
    def warmup_steps(self):
        return self.meta.get("warmup_steps") or 0

    @property
    def max_horizon(self):
        return self.horizons[-1] if self.horizons else 0


class JourneyAgent(object):
    """Watches chosen routes and decides, on every forecast, whether to interrupt the traveller.

    ``graph_provider`` returns the road graph (the web app builds it lazily). ``producer`` is
    anything with ``send(value, key=None)``; without one the agent simply does not publish.
    """

    def __init__(self, graph_provider, memory=None, producer=None, free_flow_provider=None):
        self.graph_provider = graph_provider
        self.memory = memory or JourneyMemory()
        self.producer = producer
        self.free_flow_provider = free_flow_provider or self._default_free_flow
        self._book = None
        self._incidents = {}
        self._latest = {}

    # ------------------------------------------------------------------ wiring
    def set_producer(self, producer):
        self.producer = producer

    @staticmethod
    def _default_free_flow():
        from webapp import live_readings
        free_flow, _ = live_readings.latest_free_flow_by_junction()
        return free_flow or {}

    @property
    def book(self):
        return self._book

    def set_book(self, book):
        if book is not None:
            self._book = book

    # ------------------------------------------------------------------ inbound traffic data
    def on_forecast(self, message):
        """A traffic.predictions envelope arrived: re-decide every watched journey."""
        book = ForecastBook.from_prediction_message(message)
        if book is None:
            return
        self._book = book
        self.evaluate_all()

    def on_congestion_event(self, message):
        """A traffic.congestion_events update arrived. Kept keyed, so the latest state wins."""
        event_id = message.get("event_id")
        if not event_id:
            return
        if message.get("status") == "resolved":
            self._incidents.pop(event_id, None)
            return
        self._incidents[event_id] = {
            "event_id": event_id,
            "status": message.get("status"),
            "severity": message.get("severity"),
            "event_type": message.get("event_type"),
            "area": message.get("area"),
            "junction_ids": [str(j) for j in (message.get("affected_junction_ids") or [])],
            "updated_at": message.get("updated_at")}

    # ------------------------------------------------------------------ registration
    def register(self, src, dst, horizon_min, segments, score, rank=1, label=None):
        """Start watching the route the traveller chose, exactly as the page drew it."""
        spine = routing.route_spine(segments)
        polyline = routing.route_polyline(segments)
        nodes, node_cum = routing.route_node_chain(segments)
        total_m = spine[-1]["cum_m"] if spine else 0.0
        card_eta = float(score.get("total_time_min") or 0.0)
        # The promise has to be measured the same way every later check will measure it.
        # The route card prices the whole drive at one horizon (+2 min), so a 12 km trip is
        # costed as if the next two minutes of traffic lasted the entire journey. The agent
        # prices each part at the forecast for when the car actually gets there. Taking the
        # card's number as the promise would make the agent announce a huge delay the instant
        # the journey starts, because it is comparing two different measurements rather than
        # two moments in time.
        baseline_eta = card_eta
        own_score = None
        if self._book is not None and spine:
            try:
                own_score = routing.score_spine(spine, self._book.lookup(0.0),
                                                free_flow_by_junction=self.free_flow_provider(),
                                                from_cum_m=0.0)
                baseline_eta = float(own_score.get("total_time_min") or card_eta)
            except Exception:
                LOG.exception("Could not price the chosen route; using the card's estimate")
        now = utc_now()
        journey = {
            "journey_id": "jr-" + uuid.uuid4().hex[:10],
            "label": label or "Chosen route",
            "status": "active",
            "created_at": iso(now),
            "chosen_at": iso(now),
            "src": {"lat": float(src["lat"]), "lon": float(src["lon"])},
            "dst": {"lat": float(dst["lat"]), "lon": float(dst["lon"])},
            "horizon_min": int(horizon_min),
            "chosen_rank_at_start": int(rank),
            "current_rank_label": int(rank),
            # The promise. Everything later is compared against this, never against last check.
            "baseline_eta_min": round(baseline_eta, 1),
            "promised_arrival": iso(now + timedelta(minutes=baseline_eta)),
            # What the route card showed, kept so the difference can be explained rather than
            # looking like a delay the agent invented.
            "card_eta_min": round(card_eta, 1),
            "baseline_pct_heavy": _round((own_score or score).get("pct_heavy")) or 0.0,
            "baseline_mean_speed_kmh": _round((own_score or score).get("mean_speed_kmh")) or 20.0,
            # The route itself.
            "spine": spine,
            "route_polyline": polyline,
            "route_nodes": nodes,
            "route_node_cum_m": node_cum,
            "total_length_m": round(float(total_m), 1),
            "total_length_km": _round(float(total_m) / 1000.0, 2),
            # Where the car is.
            "position_cum_m": 0.0,
            "position_source": "not_started",
            "travelled_before_m": 0.0,
            "sim_elapsed_min": 0.0,
            "sim": {"active": False, "minutes_per_second": None, "started_at": None},
            # Bookkeeping so one noisy forecast cannot swing a car.
            "suggested": None,
            "pending_alternative": None,
            "switch_suggestions": 0,
            "switch_count": 0,
            "preferences": dict(JourneyMemory.PREFERENCE_DEFAULTS),
            "tick": 0,
            "forecast_poll_id": None,
            "elapsed_at_forecast": 0.0}
        self.memory.remember(journey)
        LOG.info("Journey %s registered: rank %d, %.1f km, promised %.1f min",
                 journey["journey_id"], rank, journey["total_length_km"], baseline_eta)
        self.evaluate(journey["journey_id"])
        return self.memory.recall(journey["journey_id"])

    # ------------------------------------------------------------------ the decision loop
    def evaluate_all(self):
        for journey_id in self.memory.journey_ids():
            try:
                self.evaluate(journey_id)
            except Exception:
                LOG.exception("Journey %s could not be evaluated", journey_id)
        self.memory.prune()

    def evaluate(self, journey_id):
        """One decision for one journey. Returns the decision, or None if it is not watched."""
        with self.memory.lock:
            journey = self.memory.view(journey_id)
            if journey is None:
                return None
            if journey.get("status") in ("arrived", "expired"):
                return self._latest.get(journey_id)
            trace = []
            journey["tick"] = int(journey.get("tick", 0)) + 1
            first_check = journey["tick"] == 1

            self._trace(trace, "recall_memory", ok=True,
                        summary="%s, promised %.1f min, %d previous check(s)"
                                % (journey_id, journey.get("baseline_eta_min") or 0.0,
                                   journey["tick"] - 1))

            book = self._book
            if book is None:
                return self._finish(journey, trace, "watch", ["agent_tool_unavailable"],
                                    "No forecast has reached the agent yet, so your route has "
                                    "not been re-checked.", {})

            progress = self._estimate_progress(journey, trace)
            if progress["arrived"]:
                return self._arrive(journey, trace, progress)
            if progress["expired"]:
                return self._expire(journey, trace, progress)

            free_flow = self._get_free_flow(trace)
            assessment = self._score_current_route(journey, book, progress, free_flow, trace)
            if assessment is None:
                return self._finish(journey, trace, "watch", ["agent_tool_unavailable"],
                                    "The agent could not re-price your route on this update, so "
                                    "nothing has changed.", progress)
            incidents = self._check_incidents(journey, progress, trace)
            return self._decide(journey, book, progress, assessment, incidents, free_flow,
                                trace, first_check)

    # ------------------------------------------------------------------ tier 0: observe
    def _trace(self, trace, tool, ok=True, summary=None, skipped=False, started=None,
               expensive=False):
        call = {"tool": tool, "ok": bool(ok) and not skipped, "skipped": bool(skipped),
                "ms": int(round((time.time() - started) * 1000)) if started else 0}
        if summary:
            call["summary"] = summary
        if expensive:
            call["expensive"] = True
        trace.append(call)
        return call

    def _journey_clock_min(self, journey):
        """Minutes on this journey's own clock: simulated when a car is being driven, else real."""
        if (journey.get("sim") or {}).get("active"):
            return float(journey.get("sim_elapsed_min") or 0.0)
        chosen = parse_iso(journey.get("chosen_at"))
        if chosen is None:
            return float(journey.get("sim_elapsed_min") or 0.0)
        elapsed = (utc_now() - chosen).total_seconds() / 60.0
        journey["sim_elapsed_min"] = round(max(0.0, elapsed), 2)
        return journey["sim_elapsed_min"]

    def _estimate_progress(self, journey, trace):
        started = time.time()
        total_m = float(journey.get("total_length_m") or 0.0)
        elapsed_min = self._journey_clock_min(journey)
        source = journey.get("position_source") or "not_started"
        position = float(journey.get("position_cum_m") or 0.0)
        if source in ("not_started", "elapsed_estimate"):
            # No car being driven and no position feed: assume they are moving at the speed the
            # route was promised at. Stated plainly in the decision rather than hidden.
            speed = max(5.0, float(journey.get("baseline_mean_speed_kmh") or 20.0))
            position = min(total_m, (elapsed_min / 60.0) * speed * 1000.0)
            source = "elapsed_estimate"
            journey["position_cum_m"] = round(position, 1)
            journey["position_source"] = source
        travelled_before = float(journey.get("travelled_before_m") or 0.0)
        remaining_m = max(0.0, total_m - position)
        denominator = max(1.0, travelled_before + total_m)
        progress_pct = round(100.0 * (travelled_before + position) / denominator, 1)
        # Overdue is measured against what the agent currently expects the journey to take, not
        # against the original promise. A journey that is genuinely 35 minutes late is exactly
        # when the traveller still needs watching; only a car that has stopped getting anywhere
        # should be given up on.
        expected_total = float(journey.get("expected_total_min")
                               or journey.get("baseline_eta_min") or 0.0)
        overdue_min = elapsed_min - expected_total
        progress = {"position_cum_m": round(position, 1), "source": source,
                    "elapsed_min": round(elapsed_min, 1), "remaining_m": round(remaining_m, 1),
                    "progress_pct": min(100.0, progress_pct),
                    "arrived": remaining_m <= ARRIVAL_REMAINING_M and total_m > 0,
                    "expired": overdue_min > ARRIVAL_GRACE_MIN}
        self._trace(trace, "estimate_progress", ok=True, started=started,
                    summary="%.2f km along, %.2f km left, %.0f%% travelled (%s)"
                            % (position / 1000.0, remaining_m / 1000.0,
                               progress["progress_pct"], source))
        return progress

    def _get_free_flow(self, trace):
        started = time.time()
        try:
            free_flow = self.free_flow_provider() or {}
        except Exception:
            LOG.exception("Free-flow lookup failed; delay will be reported as unknown")
            self._trace(trace, "get_free_flow", ok=False, started=started,
                        summary="unavailable, so delay cannot be measured")
            return {}
        self._trace(trace, "get_free_flow", ok=True, started=started,
                    summary="%d junction baseline(s)" % len(free_flow))
        return free_flow

    # ------------------------------------------------------------------ tier 1: assess
    def _base_offset(self, journey, book):
        """How far the journey clock has run on past the forecast currently in hand."""
        if journey.get("forecast_poll_id") != book.poll_id:
            journey["forecast_poll_id"] = book.poll_id
            journey["elapsed_at_forecast"] = float(journey.get("sim_elapsed_min") or 0.0)
        return max(0.0, float(journey.get("sim_elapsed_min") or 0.0)
                   - float(journey.get("elapsed_at_forecast") or 0.0))

    def _score_current_route(self, journey, book, progress, free_flow, trace):
        started = time.time()
        base_offset = self._base_offset(journey, book)
        try:
            remaining = routing.score_spine(journey.get("spine") or [], book.lookup(base_offset),
                                            free_flow_by_junction=free_flow,
                                            from_cum_m=progress["position_cum_m"])
        except Exception:
            LOG.exception("Re-scoring journey %s failed", journey.get("journey_id"))
            self._trace(trace, "score_current_route", ok=False, started=started,
                        summary="the stored route could not be priced")
            return None
        elapsed = progress["elapsed_min"]
        remaining_eta = float(remaining.get("total_time_min") or 0.0)
        total_eta = elapsed + remaining_eta
        baseline = float(journey.get("baseline_eta_min") or 0.0)
        assessment = {
            "base_offset_min": round(base_offset, 1),
            "remaining": remaining,
            "remaining_eta_min": round(remaining_eta, 1),
            "total_eta_min": round(total_eta, 1),
            # The whole point: later than promised, measured on the clock.
            "eta_delta_min": round(total_eta - baseline, 1),
            "projected_arrival": iso(parse_iso(journey["chosen_at"])
                                     + timedelta(minutes=total_eta)),
            "heavy_jump_points": round(float(remaining.get("pct_heavy") or 0.0)
                                       - float(journey.get("baseline_pct_heavy") or 0.0), 1)}
        # What the agent now expects the whole journey to take, which is what the grace period
        # for giving up on a journey is measured against.
        journey["expected_total_min"] = assessment["total_eta_min"]
        self._trace(trace, "score_current_route", ok=True, started=started,
                    summary="%.1f min left over %.2f km, %.0f%% of the road ahead priced, "
                            "due %+.1f min against the promise"
                            % (remaining_eta, float(remaining.get("total_length_km") or 0.0),
                               float(remaining.get("route_coverage_pct") or 0.0),
                               assessment["eta_delta_min"]))
        return assessment

    def _remaining_junctions(self, journey, position_cum_m):
        return {entry.get("junction_id") for entry in (journey.get("spine") or [])
                if float(entry.get("cum_m", 0.0)) >= position_cum_m}

    def _check_incidents(self, journey, progress, trace):
        started = time.time()
        ahead = self._remaining_junctions(journey, progress["position_cum_m"])
        active, watching = [], []
        for event in self._incidents.values():
            if not ahead.intersection(event.get("junction_ids") or []):
                continue
            if event.get("status") in ACTIVE_INCIDENT_STATES:
                active.append(event)
            elif event.get("status") in WATCH_INCIDENT_STATES:
                watching.append(event)
        self._trace(trace, "check_incidents_on_route", ok=True, started=started,
                    summary="%d reported and %d watched on the road ahead"
                            % (len(active), len(watching)))
        return {"active": active, "watch": watching,
                "ids": [event["event_id"] for event in active + watching]}

    # ------------------------------------------------------------------ tier 2: explore
    def _find_alternatives(self, journey, book, progress, assessment, free_flow, trace):
        """Re-run the OpenStreetMap search from the car's position. The expensive step."""
        started = time.time()
        nodes = journey.get("route_nodes") or []
        node_cum = journey.get("route_node_cum_m") or []
        if len(nodes) < 2 or len(nodes) != len(node_cum):
            self._trace(trace, "find_alternatives", ok=False, started=started, expensive=True,
                        summary="this route has no stored road nodes to search from")
            return []
        position = progress["position_cum_m"]
        # Start from the next junction AHEAD of the car. Starting from the nearest node could put
        # the search behind the car, which is how a reroute ends up asking for a U-turn.
        start_index = next((i for i, cum in enumerate(node_cum) if cum >= position), None)
        if start_index is None or start_index >= len(nodes) - 1:
            self._trace(trace, "find_alternatives", ok=True, started=started, expensive=True,
                        summary="too near the destination to search")
            return []
        start_node = nodes[start_index]
        start_cum = float(node_cum[start_index])
        driven = set(nodes[:start_index])
        remaining_nodes = nodes[start_index:]
        remaining_pairs = list(zip(remaining_nodes, remaining_nodes[1:]))

        base_offset = assessment["base_offset_min"]
        lookup = book.lookup(base_offset)
        try:
            graph = self.graph_provider()
            start_lat = graph.nodes[start_node]["y"]
            start_lon = graph.nodes[start_node]["x"]
            current_tail = routing.score_spine(journey.get("spine") or [], lookup,
                                               free_flow_by_junction=free_flow,
                                               from_cum_m=start_cum)
            candidate_edges = routing.find_candidate_paths(
                graph, (start_lat, start_lon),
                (journey["dst"]["lat"], journey["dst"]["lon"]), k=CANDIDATE_PATHS)
        except routing.RoutingError as error:
            self._trace(trace, "find_alternatives", ok=False, started=started, expensive=True,
                        summary="no route could be searched from here: %s" % error)
            return []
        except Exception:
            LOG.exception("Alternatives search failed for %s", journey.get("journey_id"))
            self._trace(trace, "find_alternatives", ok=False, started=started, expensive=True,
                        summary="the road graph could not be searched")
            return []

        # The short hop from the car to that junction is common to every option, so it is added
        # to all of them and cancels out of the comparison.
        tail_speed = max(5.0, float(current_tail.get("mean_speed_kmh") or 0.0) or 20.0)
        lead_in_m = max(0.0, start_cum - position)
        lead_in_min = (lead_in_m / 1000.0) / tail_speed * 60.0
        current_tail_min = float(current_tail.get("total_time_min") or 0.0)

        alternatives, rejected = [], {"same_road": 0, "u_turn": 0, "slower": 0}
        for edges in candidate_edges:
            candidate_nodes = [edges[0]["u"]] + [edge["v"] for edge in edges]
            if routing.shared_fraction(graph, candidate_nodes,
                                       remaining_nodes) > SAME_ROUTE_OVERLAP:
                rejected["same_road"] += 1
                continue
            if self._backtracks(edges, driven):
                rejected["u_turn"] += 1
                continue
            segments = routing.colorize_path(edges, book.node_list(base_offset),
                                             free_flow_by_junction=free_flow)
            spine = routing.route_spine(segments)
            score = routing.score_spine(spine, lookup, free_flow_by_junction=free_flow,
                                        from_cum_m=0.0)
            alt_min = float(score.get("total_time_min") or 0.0)
            gain = current_tail_min - alt_min
            if gain <= 0:
                rejected["slower"] += 1
                continue
            divergence_m = self._divergence_m(edges, remaining_pairs)
            warning_m = lead_in_m + divergence_m
            warning_min = lead_in_min + (divergence_m / 1000.0) / tail_speed * 60.0
            alt_nodes, alt_cum = routing.route_node_chain(segments)
            alternatives.append({
                "signature": _signature(candidate_nodes),
                "eta_min": round(progress["elapsed_min"] + lead_in_min + alt_min, 1),
                "remaining_eta_min": round(lead_in_min + alt_min, 1),
                "gain_min": round(gain, 1),
                "gain_pct": round(100.0 * gain / max(current_tail_min, 0.1), 1),
                "total_length_km": _round(score.get("total_length_km"), 2),
                "pct_heavy": _round(score.get("pct_heavy")),
                "route_coverage_pct": _round(score.get("route_coverage_pct")),
                "warning_distance_m": round(warning_m, 1),
                "warning_time_min": round(warning_min, 1),
                "turn_off_after_km": _round(warning_m / 1000.0, 2),
                "start_node": start_node,
                "start_cum_m": round(start_cum, 1),
                "lead_in_m": round(lead_in_m, 1),
                # Kept for the switch; stripped before anything is published.
                "segments": segments, "spine": spine, "route_nodes": alt_nodes,
                "route_node_cum_m": alt_cum, "score": {k: v for k, v in score.items()}})
        alternatives.sort(key=lambda item: item["gain_min"], reverse=True)
        for position_index, alternative in enumerate(alternatives, start=1):
            alternative["rank"] = position_index
        self._trace(trace, "find_alternatives", ok=True, started=started, expensive=True,
                    summary="%d searched, %d usable (%d were your own road, %d would double "
                            "back, %d no faster)"
                            % (len(candidate_edges), len(alternatives), rejected["same_road"],
                               rejected["u_turn"], rejected["slower"]))
        return alternatives

    @staticmethod
    def _backtracks(edges, driven_nodes):
        """True when a candidate re-uses road already driven within its first few hundred metres."""
        run = 0.0
        for edge in edges:
            if edge["v"] in driven_nodes or edge["u"] in driven_nodes:
                return True
            run += float(edge.get("length_m") or 0.0)
            if run > BACKTRACK_WINDOW_M:
                return False
        return False

    @staticmethod
    def _divergence_m(edges, remaining_pairs):
        """How far along before a candidate leaves the current route."""
        run = 0.0
        for index, edge in enumerate(edges):
            if index >= len(remaining_pairs) or (edge["u"], edge["v"]) != remaining_pairs[index]:
                return run
            run += float(edge.get("length_m") or 0.0)
        return run

    # ------------------------------------------------------------------ tier 3: decide
    @staticmethod
    def _threshold_scale(journey):
        dismissed = int((journey.get("preferences") or {}).get("dismissed", 0) or 0)
        return min(MAX_DISMISSAL_SCALE, DISMISSAL_FACTOR ** dismissed) if dismissed else 1.0

    @staticmethod
    def _confidence(book, remaining):
        score = 0
        sensor = float(book.sensor_coverage_pct or 0.0)
        route = float(remaining.get("route_coverage_pct") or 0.0)
        score += 1 if sensor >= 90.0 else (-1 if sensor < MIN_COVERAGE_PCT else 0)
        score += 1 if route >= 90.0 else (-1 if route < MIN_COVERAGE_PCT else 0)
        if book.warmup_steps:
            score -= 1
        if score >= 2:
            return "high"
        return "low" if score <= 0 else "medium"

    def _decide(self, journey, book, progress, assessment, incidents, free_flow, trace,
                first_check):
        remaining = assessment["remaining"]
        delta = assessment["eta_delta_min"]
        remaining_eta = assessment["remaining_eta_min"]
        remaining_m = progress["remaining_m"]
        scale = self._threshold_scale(journey)
        confidence = self._confidence(book, remaining)
        preferences = self.memory._preference_block(journey)
        evidence = []
        if first_check:
            evidence.append("journey_registered")

        late = delta >= TRIGGER_LATE_MIN * scale
        heavier = assessment["heavy_jump_points"] >= TRIGGER_HEAVY_POINTS
        if late:
            evidence.append("arrival_later_than_when_route_was_chosen")
        if heavier:
            evidence.append("more_of_the_route_is_now_heavy")
        if incidents["active"]:
            evidence.append("congestion_event_on_remaining_route")
        elif incidents["watch"]:
            evidence.append("congestion_watch_on_remaining_route")

        endgame = remaining_eta < ENDGAME_ETA_MIN or remaining_m < ENDGAME_DISTANCE_M
        triggered = late or heavier or bool(incidents["active"])

        alternatives = []
        if endgame:
            self._trace(trace, "find_alternatives", skipped=True,
                        summary="too close to the destination to be worth rerouting")
            if triggered:
                evidence.append("too_close_to_the_destination_to_reroute")
        elif not triggered:
            self._trace(trace, "find_alternatives", skipped=True,
                        summary="your route is running as promised, so the road graph was not "
                                "searched")
        else:
            alternatives = self._find_alternatives(journey, book, progress, assessment,
                                                   free_flow, trace)

        gain_needed = max(MIN_GAIN_MIN, MIN_GAIN_SHARE * remaining_eta) * scale
        if incidents["active"]:
            gain_needed = min(gain_needed, INCIDENT_GAIN_MIN * scale)

        usable = [alternative for alternative in alternatives
                  if alternative["gain_min"] >= gain_needed
                  and alternative["warning_distance_m"] >= WARNING_DISTANCE_M
                  and alternative["warning_time_min"] >= WARNING_TIME_MIN]
        passed = {alternative["signature"] for alternative in usable}
        near_miss = [alternative for alternative in alternatives
                     if alternative["signature"] not in passed]
        best = usable[0] if usable else None

        action = "hold"
        chosen = None
        if best is not None:
            signature = best["signature"]
            previous = journey.get("suggested") or {}
            streak = int(previous.get("streak", 0)) + 1 if previous.get(
                "signature") == signature else 1
            journey["suggested"] = {"signature": signature, "streak": streak,
                                    "gain_min": best["gain_min"]}
            evidence.append("alternative_saves_%d_min" % int(round(best["gain_min"])))
            last_offer = preferences.get("last_switch_tick")
            cooling = (last_offer is not None
                       and journey["tick"] - int(last_offer) < ALERT_COOLDOWN_TICKS)
            # Normally an alternative has to win twice running, so one noisy forecast cannot
            # swing a car. But waiting costs distance: if the turn-off would be behind the car
            # by the next check, waiting means never offering it at all. So when it is now or
            # never, the agent speaks on the first check.
            step_min = float((book.meta or {}).get("step_minutes") or 2.0)
            # The speed where the car actually is, not the average of the whole road ahead: a car
            # doing 42 km/h towards a jam covers far more ground before the next check than the
            # jammed average suggests.
            here_kmh = self.road_speed_kmh(journey, progress["position_cum_m"],
                                           assessment["base_offset_min"])
            reach_m = here_kmh * 1000.0 / 60.0 * step_min
            last_chance = best["warning_distance_m"] <= reach_m
            if last_chance:
                evidence.append("last_chance_to_take_this_turn_off")
            if streak < SWITCH_STREAK and not last_chance:
                action = "watch"
                evidence.append("waiting_for_the_alternative_to_hold_another_poll")
            elif confidence == "low":
                action = "watch"
                evidence.append("confidence_too_low_to_recommend_a_switch")
            elif cooling and not last_chance:
                # The cooldown is there so the agent does not nag. It must never be the reason a
                # genuinely better road goes unmentioned on the one check where it can still be
                # taken, so a last chance overrides it.
                action = "watch"
                evidence.append("held_back_by_notification_cooldown")
            elif int(journey.get("switch_suggestions", 0)) >= MAX_SWITCH_SUGGESTIONS:
                action = "watch"
                evidence.append("already_suggested_enough_reroutes_on_this_journey")
            else:
                action = "recommend_switch"
                chosen = best
                if streak >= SWITCH_STREAK:
                    evidence.append("alternative_better_for_%d_consecutive_polls" % streak)
                journey["pending_alternative"] = best
                journey["switch_suggestions"] = int(journey.get("switch_suggestions", 0)) + 1
                preferences["last_switch_tick"] = journey["tick"]
        else:
            journey["suggested"] = None
            journey["pending_alternative"] = None
            if near_miss:
                closest = near_miss[0]
                if closest["gain_min"] < gain_needed:
                    evidence.append("no_alternative_saves_enough_to_be_worth_it")
                else:
                    evidence.append("the_turn_off_is_too_close_to_warn_you_in_time")
            last_notice = preferences.get("last_delay_tick")
            cooling = (last_notice is not None
                       and journey["tick"] - int(last_notice) < ALERT_COOLDOWN_TICKS)
            # Say "you are running late" once, not every few minutes for the rest of the trip.
            # It is repeated only when the delay has grown materially worse since last time.
            told = journey.get("last_notified_delta_min")
            worth_repeating = (told is None
                               or delta >= float(told) + NOTIFY_LATE_MIN * scale)
            if delta >= NOTIFY_LATE_MIN * scale and not cooling and worth_repeating:
                action = "notify_delay"
                preferences["last_delay_tick"] = journey["tick"]
                journey["last_notified_delta_min"] = delta
            elif triggered:
                action = "watch"
            else:
                action = "hold"
        if scale > 1.0:
            evidence.append("thresholds_raised_after_%d_dismissals"
                            % int(preferences.get("dismissed", 0)))

        rationale = self._rationale(action, journey, assessment, progress, chosen, incidents,
                                   gain_needed)
        if first_check:
            card = journey.get("card_eta_min")
            promise = float(journey.get("baseline_eta_min") or 0.0)
            if card is not None and abs(promise - float(card)) >= 2.0:
                rationale = (
                    "Watching this route. The route list showed %.0f min, which prices the whole "
                    "drive at the next forecast; allowing for the traffic forecast for when you "
                    "actually reach each stretch, expect about %.0f min. That is the figure this "
                    "agent will hold itself to. %s" % (float(card), promise, rationale))
        decision = self._finish(journey, trace, action, evidence, rationale, progress,
                                assessment=assessment, alternative=chosen, confidence=confidence,
                                incidents=incidents, book=book,
                                gain_needed_min=round(gain_needed, 1))
        return decision

    @staticmethod
    def _rationale(action, journey, assessment, progress, alternative, incidents, gain_needed):
        delta = assessment["eta_delta_min"]
        remaining_eta = assessment["remaining_eta_min"]
        if abs(delta) < 1:
            late_phrase = "within a minute of what you were promised"
        else:
            late_phrase = ("about %.0f min %s than when you chose this route"
                           % (abs(delta), "later" if delta > 0 else "earlier"))
        if action == "recommend_switch" and alternative:
            return ("A different route is now faster for the part of your journey that is left: "
                    "about %.0f min saved, turning off in about %.1f km. On your current route "
                    "you are due to arrive %s."
                    % (alternative["gain_min"], alternative["turn_off_after_km"] or 0.0,
                       late_phrase))
        if action == "notify_delay":
            jam = (" A jam is reported on the road ahead." if incidents["active"] else "")
            return ("Your route is now %s, but it is still the best way there, so there is "
                    "nothing to change.%s" % (late_phrase, jam))
        if action == "watch":
            return ("Keeping an eye on your route: %.0f min left and you are %s. Nothing worth "
                    "interrupting you for yet." % (remaining_eta, late_phrase))
        return ("Your route still holds: %.0f min left, %s."
                % (remaining_eta, late_phrase))

    # ------------------------------------------------------------------ tier 4: reflect
    def _finish(self, journey, trace, action, evidence, rationale, progress, assessment=None,
                alternative=None, confidence="medium", incidents=None, book=None,
                gain_needed_min=None, extra=None):
        started = time.time()
        remaining = (assessment or {}).get("remaining") or {}
        decision = {
            "record_type": "journey_event",
            "schema_version": 1,
            "journey_id": journey["journey_id"],
            "poll_id": (book.poll_id if book is not None else journey.get("forecast_poll_id")),
            "tick": journey.get("tick"),
            "action": action,
            "status": journey.get("status", "active"),
            "label": journey.get("label"),
            # "Now" and "when chosen" are both whole-journey minutes, so their difference is the
            # number that matters: how much later than promised they are due to arrive.
            "eta_min": (assessment or {}).get("total_eta_min"),
            "baseline_eta_min": journey.get("baseline_eta_min"),
            "eta_delta_min": (assessment or {}).get("eta_delta_min"),
            "remaining_eta_min": (assessment or {}).get("remaining_eta_min"),
            "promised_arrival": journey.get("promised_arrival"),
            "projected_arrival": (assessment or {}).get("projected_arrival"),
            "remaining_km": _round(progress.get("remaining_m", 0.0) / 1000.0, 2),
            "progress_pct": progress.get("progress_pct"),
            "progress_source": progress.get("source"),
            "elapsed_min": progress.get("elapsed_min"),
            "pct_heavy": remaining.get("pct_heavy"),
            "delay_min": remaining.get("delay_min"),
            "route_coverage_pct": remaining.get("route_coverage_pct"),
            "forecast_offset_min": (assessment or {}).get("base_offset_min"),
            "incidents_ahead": len((incidents or {}).get("active") or []),
            "incident_ids": (incidents or {}).get("ids") or [],
            "gain_needed_min": gain_needed_min,
            "confidence": confidence,
            "rationale": rationale,
            "evidence": evidence,
            "alternative": alternative,
            "tool_calls": trace,
            "sensor_coverage_pct": (book.sensor_coverage_pct if book is not None else None),
            "switch_count": journey.get("switch_count", 0),
            "updated_at": iso(utc_now())}
        if extra:
            decision.update(extra)
        self._trace(trace, "remember_decision", ok=True, started=started, summary=action)
        self._latest[journey["journey_id"]] = decision
        self.memory.record_decision(journey["journey_id"], decision)
        self.memory.touch()
        self._publish(decision)
        return decision

    def _publish(self, decision):
        if self.producer is None:
            return
        try:
            self.producer.send(slim_decision(decision), key=decision["journey_id"])
        except Exception:
            LOG.exception("Could not publish the decision for %s", decision.get("journey_id"))

    # ------------------------------------------------------------------ end of a journey
    def _arrive(self, journey, trace, progress):
        journey["status"] = "arrived"
        (journey.setdefault("sim", {}))["active"] = False
        achieved = progress["elapsed_min"]
        promised = float(journey.get("baseline_eta_min") or 0.0)
        closed = self.memory.close_switches(journey["journey_id"], achieved)
        rationale = ("You have reached the destination. The journey took about %.0f min against "
                     "the %.0f min promised when you chose this route." % (achieved, promised))
        if closed:
            rationale += (" %d switch(es) on this journey have been scored in the ledger."
                          % len(closed))
        return self._finish(journey, trace, "arrived", ["journey_arrived"], rationale, progress,
                            extra={"achieved_total_min": round(achieved, 1),
                                   "switches_scored": len(closed)})

    def _expire(self, journey, trace, progress):
        journey["status"] = "expired"
        (journey.setdefault("sim", {}))["active"] = False
        return self._finish(journey, trace, "expired", ["journey_expired"],
                            "This journey is well past its expected arrival, so it is no longer "
                            "being watched.", progress)

    # ------------------------------------------------------------------ the HTTP surface
    def latest(self, journey_id):
        decision = self._latest.get(journey_id)
        return decision if decision is not None else self.memory.last_decision(journey_id)

    def feedback(self, journey_id, response):
        return self.memory.record_feedback(journey_id, response)

    def end(self, journey_id):
        self._latest.pop(journey_id, None)
        return self.memory.forget(journey_id)

    def adopt_alternative(self, journey_id):
        """Take the recommended alternative: it becomes the route, and the new promise.

        The switch is written to the ledger here, with the gain claimed at the time. What it
        actually saved is filled in when the journey arrives.
        """
        with self.memory.lock:
            journey = self.memory.view(journey_id)
            if journey is None:
                return None
            alternative = journey.get("pending_alternative")
            if not alternative:
                return None
            decision = self._latest.get(journey_id) or {}
            before_eta = float(decision.get("eta_min")
                               or journey.get("baseline_eta_min") or 0.0)
            elapsed = float(journey.get("sim_elapsed_min") or 0.0)
            spine = alternative["spine"]
            total_m = spine[-1]["cum_m"] if spine else 0.0
            # Distance already covered stays on the books, so progress keeps counting up across
            # a switch instead of jumping back to zero.
            journey["travelled_before_m"] = round(
                float(journey.get("travelled_before_m") or 0.0)
                + float(journey.get("position_cum_m") or 0.0), 1)
            driving = bool((journey.get("sim") or {}).get("active"))
            # The new route begins at the junction the search started from, which is a short way
            # ahead of the car, so position restarts at zero on the new road.
            journey.update({
                "spine": spine,
                "route_polyline": routing.route_polyline(alternative["segments"]),
                "route_nodes": alternative["route_nodes"],
                "route_node_cum_m": alternative["route_node_cum_m"],
                "total_length_m": round(float(total_m), 1),
                "total_length_km": _round(float(total_m) / 1000.0, 2),
                "position_cum_m": 0.0,
                "position_source": "simulated_car" if driving else "elapsed_estimate",
                # The new promise. Later checks are measured against this, not the old route.
                "baseline_eta_min": round(float(alternative["eta_min"]), 1),
                "baseline_pct_heavy": alternative.get("pct_heavy") or 0.0,
                "baseline_mean_speed_kmh": _round(
                    (alternative.get("score") or {}).get("mean_speed_kmh")) or 20.0,
                "promised_arrival": iso(parse_iso(journey["chosen_at"])
                                        + timedelta(minutes=float(alternative["eta_min"]))),
                "suggested": None,
                "pending_alternative": None,
                "switch_count": int(journey.get("switch_count", 0)) + 1})
            preferences = self.memory._preference_block(journey)
            preferences["accepted"] = int(preferences.get("accepted", 0)) + 1
            preferences["switches_taken"] = int(preferences.get("switches_taken", 0)) + 1
            preferences["last_switch_tick"] = journey.get("tick")
            entry = {
                "journey_id": journey_id,
                "switched_at": iso(utc_now()),
                "tick": journey.get("tick"),
                "poll_id": decision.get("poll_id"),
                "switch_number": journey["switch_count"],
                "elapsed_min": round(elapsed, 1),
                "position_km_into_trip": _round(journey["travelled_before_m"] / 1000.0, 2),
                "progress_pct": decision.get("progress_pct"),
                "remaining_km_at_switch": decision.get("remaining_km"),
                "eta_before_min": round(before_eta, 1),
                "eta_after_min": round(float(alternative["eta_min"]), 1),
                "gain_min": alternative.get("gain_min"),
                "gain_pct": alternative.get("gain_pct"),
                "turn_off_after_km": alternative.get("turn_off_after_km"),
                "new_route_length_km": alternative.get("total_length_km"),
                "promised_arrival_before": decision.get("promised_arrival"),
                "promised_arrival_after": journey["promised_arrival"],
                "confidence": decision.get("confidence"),
                "reason": decision.get("rationale"),
                "evidence": decision.get("evidence") or [],
                "achieved_total_min": None,
                "gain_realised_min": None}
            self.memory.record_switch(entry)
            self.memory.touch()
            LOG.info("Journey %s switched: %.1f min -> %.1f min (claimed %.1f min saved)",
                     journey_id, before_eta, float(alternative["eta_min"]),
                     alternative.get("gain_min") or 0.0)
            self.evaluate(journey_id)
            return self.memory.recall(journey_id)

    def snapshot(self):
        journeys = []
        for journey in self.memory.all_journeys():
            summary = {key: value for key, value in journey.items()
                       if key not in ("spine", "route_polyline", "route_nodes",
                                      "route_node_cum_m", "pending_alternative")}
            summary["decision"] = slim_decision(self.latest(journey["journey_id"]))
            journeys.append(summary)
        return {"journey_count": len(journeys), "journeys": journeys,
                "incidents_tracked": len(self._incidents),
                "forecast_poll_id": self._book.poll_id if self._book else None,
                "forecast_horizons": self._book.horizons if self._book else [],
                "switches": self.memory.switches()}

    # ------------------------------------------------------------------ for the simulated car
    def road_speed_kmh(self, journey, cum_m, base_offset_min=0.0):
        """Predicted speed on the stretch of road the car is on, for the simulator to drive at."""
        book = self._book
        spine = journey.get("spine") or []
        if book is None or not spine:
            return float(journey.get("baseline_mean_speed_kmh") or 20.0)
        entry = next((item for item in spine if float(item.get("cum_m", 0.0)) >= cum_m),
                     spine[-1])
        node = book.lookup(base_offset_min)(entry.get("junction_id"), 0.0)
        speed = (node or {}).get("predicted_kmh")
        if speed is None:
            return float(journey.get("baseline_mean_speed_kmh") or 20.0)
        return max(routing.MIN_SPEED_KMH, float(speed))

    def base_offset_for(self, journey):
        book = self._book
        if book is None:
            return 0.0
        return max(0.0, float(journey.get("sim_elapsed_min") or 0.0)
                   - float(journey.get("elapsed_at_forecast") or 0.0))
