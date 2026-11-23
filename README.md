# Bangalore real-time traffic forecast

The project polls TomTom every **2 minutes**, writes readings to Kafka, consumes each completed
poll before running the DCRNN, and publishes forecasts back to Kafka. A feature aggregator joins
readings and forecasts per poll and publishes area-wise statistics. A congestion-event agent also
joins the raw readings and forecasts, publishes event updates to Kafka, and writes a latest-event
snapshot for the map. A journey intelligence agent watches the route a traveller actually selected
and decides, on every new forecast, whether that recommendation still holds; **Start journey** puts
a simulated car on the route, moving at the speeds the model predicts, and the agent alerts the
traveller mid-journey when a faster route exists from where they are. The map and routing page uses
the HTTP API and includes an opt-in incidents panel with separate map halos.

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
- `traffic.journey_events`: keyed decisions of the journey intelligence agent, one per watched
  journey per poll. Also compacted, so the latest decision per journey can be rebuilt.

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

### Journey intelligence agent

The congestion-event agent describes the *network*. The journey agent (`webapp/journey_agent.py`)
watches **one traveller's chosen route** and answers a different question: *is the route this
person selected still the right one, and is the change worth interrupting them about?*

There is no agent framework. A plain dictionary of tools, a loop that calls them in a fixed order,
and a record of every call that is published with the decision and shown on the page. Nothing in
it asks a language model anything; every threshold is a number in `journey_agent.py`.

#### The comparison that makes it work

Minutes-remaining always falls as you drive, so a journey can rot from 32 to 48 minutes without
any single check ever seeing a rise. The agent therefore freezes one number when the traveller
presses **Start journey** — the arrival time it promised them — and every later check asks only:
*are you now due to arrive later than that?*

That promise is the agent's **own** pricing of the route, not the figure on the route card. The
card prices the whole drive at a single horizon (`+2 min`), so a 12 km trip is costed as if the
next two minutes of traffic lasted the entire journey; `routing.score_spine` prices each stretch
at the forecast for when the car actually reaches it. Taking the card's number as the promise made
the agent announce a 56-minute delay the instant the journey started, because it was comparing two
different *measurements* rather than two *moments*. The first decision now states both figures and
says which one it will hold itself to (`card_eta_min` on the journey).

#### Where it runs

In the web-app process, not the pipeline. Deciding whether a better route exists means re-running
the road-graph search, and the road graph, `networkx` and `osmnx` exist only in the web app; the
pipeline (Python 3.7 + TensorFlow) could not do it. It is still Kafka-native: `webapp/journey_runtime.py`
consumes `traffic.predictions` on group `traffic-journey` and `traffic.congestion_events` on group
`traffic-journey-incidents` (from the beginning, because that topic is compacted), and publishes to
`traffic.journey_events`. Those groups are new, so the model (`traffic-model`), dashboard
(`traffic-dashboard`), feature and incident groups keep their own offsets and are untouched.

With no `KAFKA_BOOTSTRAP_SERVERS` set (plain `python run_webapp.py`) the same agent is fed by
polling the model service over HTTP, so the decision logic has exactly one code path. Either way
the runtime fetches a full set of horizons over HTTP at startup, so a journey begun in the first
two minutes is judged against real predictions instead of waiting for the next poll.
`GET /api/journey-runtime` reports which way it is being fed.

#### What it does on each check

| Tier | Step | Tools |
| --- | --- | --- |
| 0 | observe | `recall_memory`, `estimate_progress`, `get_free_flow` |
| 1 | assess | `score_current_route` on the **remaining** route, `check_incidents_on_route` |
| 2 | explore | `find_alternatives` &mdash; **only when a trigger has already fired** |
| 3 | decide | `hold` / `watch` / `notify_delay` / `recommend_switch` / `arrived` / `expired` |
| 4 | reflect | `remember_decision` |

The road-graph search is the expensive step (60&ndash;140 ms on the real Bangalore graph), so it is
gated: an ordinary check re-prices the stored route and stops. A tool that is deliberately not
called is recorded as skipped **with the reason**, and a tool that fails degrades that check to
`watch` instead of raising. The whole trace is published with the decision and shown on the page
under **What the agent did on this update**.

#### Searching for alternatives

The search runs from the car's current position to the destination, with `k=5` rather than the
three the map shows. Three things are then true of every candidate it will offer:

* **It is not your own road.** A candidate sharing more than 80% of its length with the route
  still ahead *is* that route (`routing.shared_fraction`, the same overlap measure the search uses
  to call two routes distinct). Searching from where you are returns the road you are on as the
  top result almost every time, so without this the banner offers you the road you are already on.
* **It does not double back.** The search starts from the next junction **ahead** of the car, never
  the nearest one, which can be the junction just passed. Any candidate re-using road already
  driven within its first 300 m is rejected.
* **There is time to act on it.** The point where it leaves the current route must be far enough
  ahead to be a usable instruction.

Routes are compared by pricing both against the same forecast from the same point, so the gain is
the difference between two answers to one question.

#### Re-scoring without the graph

At registration the route is compressed to a *spine* (`routing.route_spine`): one entry per nearest
junction with its length and cumulative distance, at most ~100 entries. `routing.score_spine`
re-prices that spine against each new forecast and reproduces `path_score` exactly for a full
route, so no numbers drift. Only the part still ahead is scored, and each part is priced at the
forecast horizon nearest to when the driver reaches it.

#### What it remembers

`webapp/journey_memory.py`, in `data/live/journey_state.json`:

* **the route itself** &mdash; the spine, the polyline, and the chain of graph nodes
  (`routing.route_node_chain`). The rank is *not* an identity: `server.py` assigns ranks by
  current predicted time, so "route 2" is a leaderboard position that reshuffles on every poll.
* **where the car is** &mdash; distance along that route.
* **the promise** &mdash; the arrival time set when the route was chosen, plus `card_eta_min`.
* **two bookkeeping fields** &mdash; the alternative suggested last time and how many checks in a
  row it has won, so one noisy forecast cannot swing a car.

Plus a short decision log (20 per journey) for the page's history and the agent's own look-back,
and the traveller's accept/dismiss counts. Journeys are pruned 180 minutes after their last update.

The **switch ledger** is separate, in `data/live/journey_switches.json`, and append-only, because
it is meant to outlive the journey:

```json
{"journey_id": "jr-ff466f71c8", "switched_at": "...", "elapsed_min": 6.0,
 "position_km_into_trip": 2.87, "progress_pct": 24.2,
 "eta_before_min": 83.5, "eta_after_min": 67.4, "gain_min": 15.8, "gain_pct": 19.2,
 "turn_off_after_km": 0.3, "new_route_length_km": 15.68,
 "reason": "A different route is now faster for ...", "evidence": [...],
 "confidence": "high",
 "achieved_total_min": 66.0, "gain_realised_min": 17.5}
```

`gain_min` is what the agent claimed at the moment of the switch; `achieved_total_min` and
`gain_realised_min` are filled in on arrival. The two columns together say whether the advice was
worth taking. `GET /api/switches` totals them.

#### Operating rules

Transparent thresholds, not calibrated probabilities &mdash; the same caveat the congestion-event
agent carries. Each is scaled up per traveller by `1.5x` per dismissal, capped at `4x`.

| Rule | Value | Why it exists |
| --- | --- | --- |
| `TRIGGER_LATE_MIN` | 3 min later than promised | noise floor before the expensive search runs |
| `TRIGGER_HEAVY_POINTS` | +15 points heavy | congestion spreading on the route |
| `MIN_GAIN_MIN` / `MIN_GAIN_SHARE` | `max(3 min, 10% of the remaining ETA)` | 4 min off a 90-min drive is noise; off a 12-min drive it is decisive. A flat threshold gets one of those wrong |
| `INCIDENT_GAIN_MIN` | 2 min | a reported jam ahead justifies a smaller promised win |
| `WARNING_DISTANCE_M` / `WARNING_TIME_MIN` | 250 m and 30 s | enough notice to act on |
| `ENDGAME_ETA_MIN` / `ENDGAME_DISTANCE_M` | 6 min or 1.5 km left | nobody reroutes two minutes from the office |
| `SWITCH_STREAK` | 2 checks | one bad forecast must not move a car |
| `ALERT_COOLDOWN_TICKS` | 3 checks | minimum spacing between interruptions |
| `MAX_SWITCH_SUGGESTIONS` | 3 per journey | a nav that nags gets switched off |
| `MIN_COVERAGE_PCT` | 70% | never reroute on guesses |
| `NOTIFY_LATE_MIN` | 5 min | tell them they are late even when no better route exists |
| `ARRIVAL_GRACE_MIN` | 30 min | stop watching this long after the arrival **currently expected** |

Four of these were set by watching simulated journeys fail, and are worth knowing about:

* **`WARNING_DISTANCE_M` started at 500 m.** A bypass saving 19 minutes sat 460 m ahead at
  42 km/h &mdash; 39 seconds of notice, more than a satnav usually gives &mdash; and the rule threw
  it away, after which the car crawled for another 42 minutes with no alternative left to offer.
  A floor that rejects usable advice is worse than no floor.
* **The confirmation streak can cost the turn-off.** Waiting one check for an alternative to prove
  itself costs distance, and the junction can be behind you by then. When the turn-off would be
  passed before the next check, `last_chance_to_take_this_turn_off` lets the agent speak on the
  first one.
* **There are two separate interruption budgets**, `last_switch_tick` and `last_delay_tick`.
  Sharing one meant a "you are running late" notice at registration silenced a switch offer worth
  35 minutes two checks later. Telling someone they are late is a different message from telling
  them about a faster road. A last chance overrides the cooldown outright.
* **A delay notice is not repeated** unless the delay has grown by another `NOTIFY_LATE_MIN`.
  Otherwise the agent announced "36 min late" every six minutes for the rest of the trip.
* **`ARRIVAL_GRACE_MIN` is measured against the arrival now expected**, not the original promise.
  Measuring from the promise abandoned a journey that was genuinely 36 minutes late 1.5 km from the
  destination &mdash; exactly when the traveller still needed watching.

Confidence (`high` / `medium` / `low`) comes from sensor coverage, warm-up steps and how much of
the route ahead had a forecast. Low confidence never produces a switch recommendation.

#### The simulated car

`webapp/journey_simulator.py` drives a watched journey along its own route. It runs in the web app,
not the browser, so a journey keeps moving through a page refresh or a closed tab, and the agent is
driven by the car rather than by whoever happens to be looking.

Two things make it honest rather than a cartoon:

* The car moves at the speed the model predicts for the road it is on, so it genuinely slows in
  traffic and the arrival times mean something.
* Time is compressed, but the forecasts are not faked. One poll already carries twelve horizons
  (`predict_bangalore.py`: `horizon_minutes`) &mdash; in the docker test cadence, +2 through
  +24 minutes &mdash; so when the car's clock says ten minutes have passed, the agent reads the
  ten-minute forecast instead of waiting ten real minutes. A 24-minute drive replays in well under
  a minute and every decision in it is made on a real DCRNN prediction.

`minutes_per_second` on `POST /api/journey` sets the compression (default `0.5`, one simulated
minute per two real seconds; `1.0` is used by the scripted run). Decisions are taken every two
*simulated* minutes, matching the model's own cadence. A simulated journey longer than the twelfth
horizon reuses the last horizon for its tail, which each decision reports through
`forecast_offset_min`. `POST /api/journey/{id}/position` still accepts a real `{lat, lon}` and
replaces the simulated position without changing any policy; with neither a car nor a feed,
progress is estimated from elapsed time and the decision says so in `progress_source`.

#### Three honest limits

* **Junction density caps how often a reroute can be found.** Speeds come from the nearest of 100
  junctions, so two roads a kilometre apart often share the same junction and therefore the same
  predicted speed. On `Marathahalli -> Silk Board`, four of the five candidate routes ran through
  the same five junctions and could not be told apart at all. The agent can only ever find an
  alternative when it differs in the junctions it passes.
* **Candidates are shortlisted by distance, not by time.** `find_candidate_paths` weights the
  search by `length` and discards anything over `MAX_DETOUR` (2x the shortest), *then* prices the
  survivors by predicted traffic. So a longer road that is flowing freely can be eliminated before
  traffic is ever considered &mdash; precisely the case this feature is for. Raising `k` to 5 widens
  the net; weighting the search by predicted travel time would fix it properly, but that changes
  the routing the main map uses and was left alone.
* **The route cards are ranked by single-horizon pricing, so "Best" may not be best.** `/api/route`
  ranks candidates with `path_score`, which prices the whole drive at one horizon. Under the
  agent's per-horizon pricing the order can differ: on `Marathahalli -> Silk Board` with the inner
  ORR jamming from +4 min, the card called route 1 fastest (17.8 min vs 19.6) while the agent
  priced route 1 at 64 min and the alternative at 30. The traveller is therefore started on a route
  the agent already considers worse, and only acts on it once a trigger fires. Ranking the cards
  with `score_spine` as well would remove the discrepancy; it was not changed here because it
  alters the main routing display, which was outside this feature's scope.

  This is also why the agent carries a second trigger. With an honest baseline, "are you later
  than promised?" stays quiet when the jam was correctly predicted from the start &mdash; the route
  was always going to be slow. What matters then is that the balance between the routes has
  shifted, which `more_of_the_route_is_now_heavy` detects.

#### Each `traffic.journey_events` message

A full state update for its `journey_id` key:

```json
{
  "record_type": "journey_event",
  "schema_version": 1,
  "journey_id": "jr-ff466f71c8",
  "poll_id": "...",
  "tick": 4,
  "action": "recommend_switch",
  "status": "active",
  "eta_min": 83.5,
  "baseline_eta_min": 73.4,
  "eta_delta_min": 10.1,
  "remaining_eta_min": 77.5,
  "promised_arrival": "...",
  "projected_arrival": "...",
  "remaining_km": 9.0,
  "progress_pct": 24.2,
  "progress_source": "simulated_car",
  "elapsed_min": 6.0,
  "pct_heavy": 61.0,
  "delay_min": 11.3,
  "route_coverage_pct": 100.0,
  "forecast_offset_min": 6.0,
  "incidents_ahead": 0,
  "gain_needed_min": 7.8,
  "confidence": "high",
  "rationale": "A different route is now faster for the part of your journey that is left ...",
  "evidence": ["arrival_later_than_when_route_was_chosen",
               "more_of_the_route_is_now_heavy",
               "alternative_saves_16_min",
               "alternative_better_for_2_consecutive_polls"],
  "alternative": {"rank": 1, "eta_min": 67.4, "gain_min": 15.8, "gain_pct": 19.2,
                  "total_length_km": 15.68, "turn_off_after_km": 0.3},
  "tool_calls": [{"tool": "score_current_route", "ok": true, "ms": 3,
                  "summary": "77.5 min left over 9.00 km, 100% of the road ahead priced, "
                             "due +10.1 min against the promise"},
                 {"tool": "find_alternatives", "ok": true, "ms": 115, "expensive": true,
                  "summary": "5 searched, 2 usable (1 were your own road, 0 would double "
                             "back, 2 no faster)"}],
  "sensor_coverage_pct": 100.0,
  "switch_count": 0,
  "updated_at": "..."
}
```

`eta_min` and `baseline_eta_min` are both whole-journey minutes, so their difference is the number
that matters. The alternative's road geometry is kept out of the Kafka message and served over HTTP
instead.

#### HTTP API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/journey` | **Start journey**: watch the chosen route (`rank`) and put a car on it (`simulate`, `minutes_per_second`) |
| `GET /api/journey/{id}` | the latest decision, the journey (with `car_position`), and the last five decisions |
| `GET /api/journey/{id}/route` | the journey's own route as a line of points, for the page to draw |
| `POST /api/journey/{id}/feedback` | `{"response": "accepted" \| "dismissed"}`; a dismissal raises the thresholds |
| `POST /api/journey/{id}/switch` | adopt the recommended alternative; returns its segments and becomes the new promise |
| `POST /api/journey/{id}/position` | optional real position feed |
| `DELETE /api/journey/{id}` | stop watching |
| `GET /api/journeys` | every watched journey and its latest decision |
| `GET /api/switches` | the switch ledger, with claimed and realised gains totalled |
| `GET /api/journey-runtime` | whether the agent is fed by Kafka or by polling |

A `404` from `GET /api/journey/{id}` tells the page the journey has been forgotten, and the page
drops its stored journey.

#### On the page

Each route card carries **Start journey**. Pressing it registers the route and starts the car; a
blue dot moves along the road with the predicted speed on its tooltip, and the journey's own route
is drawn as a separate blue line. A banner above the route list shows the current decision: the ETA
now against when the route was chosen, the difference, how much is left, progress, any alternative,
the reasons in plain words, and the tool trace. `Switch to the faster route`, `Keep my route`
(recorded as a dismissal) and `End journey` appear as appropriate, and a panel below lists the
switches taken with the minutes each claimed and actually saved.

While a journey is running the page polls `GET /api/journey/{id}` every 1.5 s, because decisions
arrive every few real seconds under time compression, and it stops recomputing candidate routes:
once a car is moving, routes from the original source are not what is being driven, and redrawing
them would keep pulling the map away from the car. The journey id is kept in `localStorage`, so a
reload restores the points, the horizon, the drawn route and the journey instead of snapping back
to route 1.

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
docker compose exec kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic traffic.journey_events --from-beginning --max-messages 1
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
- `webapp`: map, forecast display, route comparison UI, the area-wise dashboard, and the
  journey intelligence agent (`traffic.journey_events`) with its simulated car and switch ledger,
  which run here because deciding whether a better route exists needs the road graph.
- `kafka`: persistent single-node KRaft broker with the five application topics
  (`traffic.raw`, `traffic.predictions`, `traffic.features`, `traffic.congestion_events`,
  `traffic.journey_events`).
