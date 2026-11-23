"""The moving car: drives a watched journey along its own route so the agent has someone to talk to.

It runs in the web app, not in the browser, so a journey keeps moving through a page refresh or a
closed tab, and the agent is driven by the car rather than by whoever happens to be looking.

Two things make it honest rather than a cartoon:

* The car moves at the speed the model predicts for the road it is on, so it genuinely slows down
  in traffic and the agent's arrival times mean something.
* Time is compressed, but the forecasts are not faked. One poll from the model already carries
  twelve forecasts -- in the docker test cadence, +2 through +24 minutes -- so when the car's own
  clock says ten minutes have passed, the agent reads the ten-minute forecast instead of waiting
  ten real minutes. A twenty-four minute drive replays in well under a minute, and every decision
  in it is made on a real DCRNN prediction.

The only limit is the twelfth horizon: a simulated journey longer than the model forecasts ahead
reuses the last horizon for the tail of the trip, which the decision reports through
``forecast_offset_min``.
"""
from __future__ import absolute_import

import logging
import threading

from webapp.journey_memory import iso, utc_now

LOG = logging.getLogger(__name__)

TICK_SECONDS = 1.0            # real seconds between movements
MINUTES_PER_SECOND = 0.5      # one simulated minute per two real seconds
DECISION_EVERY_SIM_MIN = 2.0  # the agent decides on the model's own cadence, in journey time
MAX_MINUTES_PER_SECOND = 10.0


class JourneySimulator(threading.Thread):
    """Advances every journey that has a car running, and asks the agent to decide on cadence."""

    def __init__(self, agent, tick_seconds=TICK_SECONDS,
                 minutes_per_second=MINUTES_PER_SECOND,
                 decision_every_sim_min=DECISION_EVERY_SIM_MIN):
        super(JourneySimulator, self).__init__(name="journey-simulator", daemon=True)
        self.agent = agent
        self.memory = agent.memory
        self.tick_seconds = float(tick_seconds)
        self.minutes_per_second = float(minutes_per_second)
        self.decision_every_sim_min = float(decision_every_sim_min)
        self._stop = threading.Event()

    # ------------------------------------------------------------------ control
    def start_journey(self, journey_id, minutes_per_second=None):
        """Put a car on this journey at its source and start it moving."""
        speed = float(minutes_per_second or self.minutes_per_second)
        speed = max(0.0, min(MAX_MINUTES_PER_SECOND, speed))
        with self.memory.lock:
            journey = self.memory.view(journey_id)
            if journey is None:
                return None
            journey["sim"] = {"active": True, "minutes_per_second": speed,
                              "tick_seconds": self.tick_seconds,
                              "started_at": iso(utc_now())}
            journey["position_cum_m"] = 0.0
            journey["position_source"] = "simulated_car"
            journey["sim_elapsed_min"] = 0.0
            journey["next_decision_at_min"] = self.decision_every_sim_min
            self.memory.touch()
        LOG.info("Journey %s: car started at %.2f simulated minutes per real second",
                 journey_id, speed)
        return self.memory.recall(journey_id)

    def stop_journey(self, journey_id):
        with self.memory.lock:
            journey = self.memory.view(journey_id)
            if journey is None:
                return None
            journey.setdefault("sim", {})["active"] = False
            self.memory.touch()
        return self.memory.recall(journey_id)

    def close(self):
        self._stop.set()

    # ------------------------------------------------------------------ the loop
    def run(self):
        LOG.info("Journey simulator running (tick %.1fs)", self.tick_seconds)
        while not self._stop.is_set():
            try:
                for journey_id in self.memory.journey_ids():
                    if self._advance(journey_id):
                        self.agent.evaluate(journey_id)
            except Exception:
                LOG.exception("Journey simulator tick failed; carrying on")
            self._stop.wait(self.tick_seconds)

    def _advance(self, journey_id):
        """Move one car one tick. Returns True when the agent should decide now."""
        with self.memory.lock:
            journey = self.memory.view(journey_id)
            if journey is None:
                return False
            sim = journey.get("sim") or {}
            if not sim.get("active"):
                return False
            if journey.get("status") != "active":
                sim["active"] = False
                return False

            advance_min = self.tick_seconds * float(
                sim.get("minutes_per_second") or self.minutes_per_second)
            if advance_min <= 0:
                return False
            total_m = float(journey.get("total_length_m") or 0.0)
            position = float(journey.get("position_cum_m") or 0.0)
            base_offset = self.agent.base_offset_for(journey)
            speed_kmh = self.agent.road_speed_kmh(journey, position, base_offset)
            position = min(total_m, position + speed_kmh * 1000.0 * (advance_min / 60.0))
            elapsed = round(float(journey.get("sim_elapsed_min") or 0.0) + advance_min, 2)

            journey["position_cum_m"] = round(position, 1)
            journey["position_source"] = "simulated_car"
            journey["sim_elapsed_min"] = elapsed
            journey["current_speed_kmh"] = round(speed_kmh, 1)
            journey["car_position"] = self._point(journey, position)

            at_end = total_m > 0 and position >= total_m - 1.0
            due = float(journey.get("next_decision_at_min") or 0.0)
            if at_end or elapsed >= due:
                journey["next_decision_at_min"] = round(elapsed + self.decision_every_sim_min, 2)
                return True
            # Positions between decisions are deliberately not written to disk every second;
            # the next decision persists them.
            return False

    @staticmethod
    def _point(journey, position_cum_m):
        from webapp import routing
        point = routing.position_at(journey.get("route_polyline") or [], position_cum_m)
        if point is None:
            return None
        return {"lat": round(point[0], 6), "lon": round(point[1], 6),
                "cum_m": round(float(position_cum_m), 1)}
