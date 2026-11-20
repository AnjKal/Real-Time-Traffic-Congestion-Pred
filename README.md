# Bangalore real-time traffic forecast

The project polls TomTom every **2 minutes**, writes readings to Kafka, consumes each completed
poll before running the DCRNN, and publishes forecasts back to Kafka. The map and routing page
continues to use the HTTP API, whose forecast state is populated by a consumer of the predictions
topic.

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

The first TomTom poll runs at the next 2-minute boundary. The first forecast is published after
that poll has been consumed from `traffic.raw`; it does not wait for 12 polls. The model still
expects a 12-step input, so missing early history is padded from the available readings and the
forecast reports its warm-up step count. Missing junction readings use the last available speed,
or that junction's reference speed when no live value has arrived yet; coverage is included with
each forecast. After 12 polls, the input window contains 12 two-minute poll slots. The DCRNN was
trained on 15-minute patterns, so the 2-minute cadence is for pipeline testing and does not
establish forecast accuracy.

## Kafka topics

Docker Compose creates these application topics before starting the pipeline:

- `traffic.raw`: one JSON reading per junction, followed by a poll-complete marker.
- `traffic.predictions`: one JSON forecast envelope per consumed poll, with all forecast horizons.

The raw consumer batches readings by poll ID, archives the completed batch to
`data/live/readings.csv`, and only then runs inference. The prediction consumer reads the forecast
envelope from `traffic.predictions` and supplies the HTTP API. Kafka consumer offsets are committed
after processing; raw offsets are committed only after a poll-complete marker is handled.

List topics and inspect the pipeline logs:

```powershell
docker compose exec kafka kafka-topics --bootstrap-server kafka:29092 --list
docker compose logs -f pipeline
```

Optionally read a few raw and prediction messages:

```powershell
docker compose exec kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic traffic.raw --from-beginning --max-messages 3
docker compose exec kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic traffic.predictions --from-beginning --max-messages 1
```

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

- `pipeline`: DCRNN, TomTom collector, raw-topic consumer, predictions producer and consumer, HTTP API.
- `webapp`: map, forecast display, and route comparison UI.
- `kafka`: persistent single-node KRaft broker with the two application topics.
