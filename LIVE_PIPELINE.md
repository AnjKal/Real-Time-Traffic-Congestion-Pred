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

The collector also appends successful TomTom readings to
`data/live/tomtom_readings_archive.csv` for storage only. It is separate from the model's
`data/live/readings.csv` archive and is not read by the model or UI.
