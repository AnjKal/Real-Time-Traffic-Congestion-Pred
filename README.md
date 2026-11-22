# Bangalore real-time traffic forecast

The project polls TomTom every **2 minutes**, writes readings to Kafka, consumes each completed
poll before running the DCRNN, and publishes forecasts back to Kafka. A feature aggregator joins
readings and forecasts per poll and publishes area-wise statistics. A congestion-event agent also
joins the raw readings and forecasts, publishes event updates to Kafka, and writes a latest-event
snapshot for the map. The map and routing page uses the HTTP API and includes an opt-in incidents
panel with separate map halos.

> **Note on nodes:** keep two different "nodes" distinct. The DCRNN models **100 traffic junctions**
> (`J001`–`J100`, one per row of `bangalore_traffic_spots.csv`). The router works on a separate
> **OSM road-graph** of intersections; each road segment borrows traffic values from its *nearest*
> junction.

## Requirements

- Docker Desktop with Docker Compose
- A TomTom API key with access to Flow Segment Data
- Kafka runs in KRaft mode; no separate Zookeeper service is required.

## First-time setup

From the repository directory, create `.env` from the example if needed, then add your key:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set `TOMTOM_API_KEY` in `.env`. Do not commit this file.

Build the web image and download the road graph once:

```powershell
docker compose build webapp
docker compose run --rm --no-deps webapp python -m webapp.osm_graph
```

## Start in the background

```powershell
docker compose up --build -d
```

Open the map at [http://localhost:8010](http://localhost:8010). The model API health endpoint is
[http://localhost:8001/health](http://localhost:8001/health).

The first TomTom poll runs at the next 2-minute boundary. Each completed poll triggers a forecast.
Although the model expects a 12-step input, missing early history is padded from the readings
available so the first forecast does not wait for 12 polls. The forecast reports its warm-up step
count. After 12 polls, the input window contains 12 two-minute poll slots. Missing junction
readings use the last available speed, or that junction's reference speed when no live value has
arrived yet; coverage is included with each forecast.

## Kafka topics

Docker Compose creates these application topics before starting the pipeline:

- `traffic.raw`: one JSON reading per junction, followed by a poll-complete marker.
- `traffic.predictions`: one JSON forecast envelope per consumed poll, with all forecast horizons.
- `traffic.features`: one JSON **area-wise** aggregation per poll (see schema below).
- `traffic.congestion_events`: keyed state updates for observed slowdowns and forecast
  congestion watches. The topic is compacted so consumers can rebuild the latest state.

The raw consumer (group `traffic-model`) batches readings by poll ID, archives the completed batch
to `data/live/readings.csv`, and only then runs inference. The prediction consumer (group
`traffic-dashboard`) reads the forecast envelope from `traffic.predictions` and supplies the HTTP
API. Kafka consumer offsets are committed after processing; raw offsets are committed only after a
poll-complete marker is handled.

The **feature aggregator** (`feature_aggregator.py`) adds two more consumers on **dedicated groups**
so it never disturbs the DCRNN raw consumer: a raw consumer (group `traffic-features-raw`) collects
each poll's TomTom `currentSpeed`/`freeFlowSpeed`, and a predictions consumer (group
`traffic-features`) joins them **by `poll_id`** with the DCRNN predicted speed and band, aggregates
by area, and publishes one `area_features` message to `traffic.features`. Duplicate predictions for
a poll already processed are ignored, and readings from different polls are never mixed. The latest
envelope is also written atomically to `data/live/features.json`, which the web app serves at
`GET /api/features` for the area-wise dashboard.

### Congestion-event agent

The incident agent has dedicated consumers on `traffic.raw` and `traffic.predictions`; it joins
them by `poll_id` without taking offsets from the model or feature aggregator. It compares real
TomTom `currentSpeed` and forecast speeds with TomTom `freeFlowSpeed`, tracks recent speed ratios,
checks nearby junctions within 2.5 km, and uses persistence and separate recovery thresholds to
avoid flickering alerts. One small change in one poll is not enough by itself; the current slowdown
must persist across two 2-minute polls or be corroborated nearby. The agent uses forecasts as soon
as they are emitted, including forecasts with padded warm-up steps, and records the actual forecast
horizon and warm-up count in each event.

Event states are `watch`, `active`, `recovering`, and `resolved`; event types describe congestion
(`forecast_congestion`, `observed_slowdown`, or `congestion_build_up`). They are not predictions of
accidents. The initial thresholds in `incident_agent.py` are transparent rules, not calibrated
probabilities. Each topic update is keyed by `event_id` and includes evidence, affected junctions,
speed ratios, coverage, freshness, and model cadence.

Each `traffic.congestion_events` message is a full state update for its event key:

```json
{
  "record_type": "congestion_event",
  "schema_version": 1,
  "event_id": "ce-j001-...",
  "event_type": "observed_slowdown | forecast_congestion | congestion_build_up",
  "status": "watch | active | recovering | resolved",
  "severity": "low | moderate | high",
  "area": "HSR Layout",
  "anchor_junction_id": "J001",
  "affected_junction_ids": ["J001", "J002"],
  "avg_current_speed_kmh": 18.4,
  "avg_free_flow_speed_kmh": 42.0,
  "avg_predicted_speed_kmh": 16.2,
  "forecast_horizon_min": 2,
  "speed_ratio_current": 0.438,
  "evidence": ["observed_speed_below_tomtom_free_flow"],
  "signal_quality": "medium",
  "sensor_coverage_pct": 96.0,
  "forecast_used": true,
  "step_minutes": 2,
  "warmup_steps": 11,
  "model_mode": "pipeline_test",
  "source_poll_id": "...",
  "first_seen": "...",
  "last_seen": "...",
  "updated_at": "..."
}
```

The agent writes the latest snapshot atomically to `data/live/incidents.json` for
`GET /api/incidents`. The map shows separate halos for active/watch events; the list opens only
after clicking **See incidents**. Locations identify modeled junction areas, not exact points on a
road.

The collector also appends successful raw TomTom readings to
`data/live/tomtom_readings_archive.csv`. This archive is storage only; model and UI code do not
read it. The model's separate `data/live/readings.csv` remains the input archive used by existing
model and route features.

### `traffic.features` message schema (`record_type: "area_features"`, `schema_version: 1`)

```
poll_id, timestamp, made_at, step_minutes, data_age_min,
sensor_coverage_pct,            # overall % of the 100 junctions with a live reading this poll
area_count,
areas: [ {
  area,                         # area name from bangalore_traffic_spots.csv
  total_junction_count,         # modeled junctions in this area
  reporting_junction_count,     # junctions with a real TomTom reading this poll
  sensor_coverage_pct,
  avg_current_speed_kmh,        # mean TomTom currentSpeed of reporting junctions (null if none)
  avg_free_flow_speed_kmh,      # mean TomTom freeFlowSpeed (null if none)
  avg_predicted_speed_kmh,      # mean DCRNN predicted speed
  congestion_distribution: { free, moderate, heavy },   # junction counts
  overall_congestion_level,     # majority band; ties resolve to the more severe band
  avg_delay_min_per_km          # mean of max(0, 60/predicted - 60/free_flow) per km (null if none)
} ]
```

List topics and inspect the pipeline logs:

```powershell
docker compose exec kafka kafka-topics --bootstrap-server kafka:29092 --list
docker compose logs -f pipeline
```

Optionally read a few raw and prediction messages:

```powershell
docker compose exec kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic traffic.raw --from-beginning --max-messages 3
docker compose exec kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic traffic.predictions --from-beginning --max-messages 1
docker compose exec kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic traffic.features --from-beginning --max-messages 1
docker compose exec kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic traffic.congestion_events --from-beginning --max-messages 1
```

## Speed metrics and the delay calculation

The UI deliberately distinguishes four different speeds, which must not be conflated:

| Metric | Source | Meaning |
| --- | --- | --- |
| **Current speed** | TomTom `currentSpeed` (`traffic.raw`) | live measured speed now |
| **Free-flow speed** | TomTom `freeFlowSpeed` (`traffic.raw`) | the road's uncongested speed, measured by TomTom |
| **Reference speed** | 90th percentile of training speeds (`reference_kmh`) | the model's *historical usual* speed — **not** TomTom free-flow |
| **Predicted speed** | DCRNN (`traffic.predictions`) | forecast speed at the chosen horizon |

**Route hover cards** show current, free-flow (TomTom), predicted, congestion band, the nearest
junction and its distance, and an estimated delay. The **delay baseline is actual TomTom free-flow**:

```
delay_min = max(0, (length_m/1000 / predicted_speed_kmh - length_m/1000 / free_flow_speed_kmh) * 60)
```

If free-flow is unavailable for a segment's junction the delay shows `N/A` — `reference_kmh` is never
substituted for it. A dashed segment and a warning appear when the nearest junction is over 3 km
away (the colour is extrapolated). The free-flow value is read read-only from `data/live/readings.csv`
(the latest valid reading per junction), so no Kafka, collector, or model code is involved.

The **area-wise dashboard** below the map lists each area's current / free-flow / predicted average
speed, congestion level, average delay per km, reporting junction count, coverage, and update time,
with stale-data and missing-data states.

## Testing

Pure-logic tests (no Kafka/model/Docker) run in the web-app Python (3.10+):

```powershell
python -m pytest test_stage2_features.py -v
```

They cover free-flow propagation, the delay formula and its handling of missing/zero/invalid
speeds, latest-free-flow extraction from the readings CSV, junction-to-area mapping, and area
feature aggregation (averages, congestion distribution, coverage, missing readings, duplicate and
cross-poll protection). The existing `test_webapp_offline.py` additionally needs `httpx`
(`pip install httpx`) because it uses Starlette's `TestClient`.

## Stop and restart

```powershell
docker compose stop
docker compose start
```

To stop and remove the containers while keeping the Kafka, readings, forecasts, and map data:

```powershell
docker compose down
```

The TomTom collector has no hard-coded daily request quota. It keeps request pacing and retries
for transient failures; TomTom can still enforce the limits associated with the configured API
key. The dashboard is available while the services run in the background. Use
`docker compose logs -f` to follow service output.

## Services

- `pipeline`: DCRNN, TomTom collector, raw-topic consumer, predictions producer and consumer, the
  feature aggregator (`traffic.features`), congestion-event agent
  (`traffic.congestion_events`), and the HTTP API.
- `webapp`: map, forecast display, route comparison UI, and the area-wise dashboard.
- `kafka`: persistent single-node KRaft broker with the four application topics
  (`traffic.raw`, `traffic.predictions`, `traffic.features`, `traffic.congestion_events`).
