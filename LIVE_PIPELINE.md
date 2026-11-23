# Live traffic pipeline

The live app now runs through Docker Compose. Follow the current setup, background startup,
topic inspection, and shutdown instructions in [README.md](README.md).

The pipeline polls at a two-minute test cadence. TomTom readings enter `traffic.raw`; the raw consumer
assembles each poll and triggers inference; forecasts enter `traffic.predictions`; and the
predictions consumer supplies the HTTP API. It emits the first forecast after the first completed
poll by padding unavailable input steps and reporting the warm-up count; the incident agent uses
that forecast immediately and records its actual horizon.

A feature aggregator (`feature_aggregator.py`) consumes `traffic.raw` and `traffic.predictions` on
dedicated consumer groups (so the DCRNN raw consumer is untouched), joins them by `poll_id`,
aggregates per area, and publishes `area_features` to `traffic.features`. The latest envelope is
written to `data/live/features.json` and served by the web app at `GET /api/features`. See the
schema and metric definitions in [README.md](README.md).

The congestion-event agent joins raw readings and forecasts on dedicated consumer groups by
`poll_id`. It emits keyed, stateful updates to `traffic.congestion_events`, writes the current
snapshot to `data/live/incidents.json`, and exposes it at `GET /api/incidents`. The map keeps
event halos visible and opens the event list from the **See incidents** button.

The journey intelligence agent runs in the **web app** process, because deciding whether a
better route exists needs the road graph that only the web app has. It consumes
`traffic.predictions` (group `traffic-journey`) and `traffic.congestion_events` (group
`traffic-journey-incidents`) and publishes one keyed decision per watched journey per poll to
`traffic.journey_events`. With no broker configured it falls back to polling the model service,
using the same decision code; `GET /api/journey-runtime` says which way it is being fed.

**Start journey** on a route card also puts a simulated car on that route
(`webapp/journey_simulator.py`), moving at the speeds the model predicts. Time is compressed by
reading the later horizons the model already publishes rather than waiting for wall-clock time, so
a journey replays in under a minute with every decision made on a real prediction. Adopted
switches are appended to `data/live/journey_switches.json` with the gain claimed at the time and,
once the journey arrives, the gain actually achieved; `GET /api/switches` totals them. The
agent's state lives in `data/live/journey_state.json`. Its thresholds, memory, tool trace, two
known limits and HTTP API are described in [README.md](README.md).

The collector also appends successful TomTom readings to
`data/live/tomtom_readings_archive.csv` for storage only. It is separate from the model's
`data/live/readings.csv` archive and is not read by the model or UI.
