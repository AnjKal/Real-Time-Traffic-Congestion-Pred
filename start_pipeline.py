"""Run the Kafka-backed live traffic pipeline.

    python start_pipeline.py --kafka-bootstrap localhost:9092

The TomTom collector only publishes to ``traffic.raw``. A raw-topic consumer assembles each poll,
archives it, and triggers inference. Forecasts are published to ``traffic.predictions``; a second
consumer supplies the HTTP API from that topic. With no interactive terminal, the process stays
alive in the background and all component logs go to stdout/stderr.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import collect_tomtom as ct
from kafka_consumer import JsonConsumer, RawPollAssembler
from kafka_producer import JsonProducer
from kafka_topics import RAW_TOPIC, PREDICTIONS_TOPIC
from model_service import ForecastEngine, create_app
from predict_bangalore import BangaloreForecaster, STEP_MINUTES

LOG = logging.getLogger("traffic_pipeline")
PRINT_LOCK = threading.Lock()


def say(text=""):
    with PRINT_LOCK:
        print(text)
        sys.stdout.flush()


def format_node(node):
    lines = ["", "%s (%s)   now %.1f km/h   usual speed %.1f km/h" % (
        node.get("name", node["junction_id"]), node["junction_id"], node["current_kmh"],
        node["reference_kmh"]),
        "forecast built from readings up to %s, data %.1f min old" % (
            node.get("input_last_slot", "unknown")[11:19], node.get("data_age_min", 0.0))]
    if node.get("warning"):
        lines.append("*** PIPELINE TEST: plumbing check, not a measure of model accuracy ***")
    if node.get("warmup_warning"):
        lines.append("*** %s ***" % node["warmup_warning"])
    for horizon in node["horizons"]:
        lines.append("  +%2d min (%s)   %5.1f km/h   %s" % (
            horizon["minutes_ahead"], horizon["forecast_for"][11:16],
            horizon["predicted_kmh"], horizon["band"]))
    return "\n".join(lines)


def progress_text(engine):
    health = engine.health()
    if health["ready"]:
        return "forecast consumed from traffic.predictions (poll %s)" % health.get("latest_poll_id")
    return health.get("not_ready_reason", "waiting for Kafka data")


def terminal_loop(engine, stop):
    say("\nType a junction name or ID (e.g. silk board, J001), 'list', or 'quit'.")
    while not stop.is_set():
        try:
            line = input("> ").strip()
        except EOFError:
            stop.wait()
            return
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            return
        if line.lower() == "list":
            for junction in engine.list_junctions():
                say("  %s  %s" % (junction["junction_id"], junction["name"] or ""))
            continue
        try:
            say(format_node(engine.node_forecast(line)))
        except KeyError as exc:
            say("  " + str(exc.args[0]))
        except ValueError as exc:
            say("  " + str(exc))
        except LookupError as exc:
            say("  No forecast yet: " + str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-min", type=int, default=2,
                        help="minutes between readings (2-minute test cadence by default)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP bind address; use 0.0.0.0 inside Docker")
    parser.add_argument("--max-polls", type=int, default=None)
    parser.add_argument("--readings", default=ct.READINGS_PATH,
                        help="CSV archive written after raw messages are consumed")
    parser.add_argument("--forecast-dir", default=None, help="where model inputs/forecasts are saved")
    parser.add_argument("--no-collect", action="store_true",
                        help="consume existing Kafka topics and serve forecasts without polling")
    parser.add_argument("--interactive", action="store_true",
                        help="also accept junction queries in this terminal")
    parser.add_argument("--kafka-bootstrap", default=os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    args = parser.parse_args()

    if args.step_min < 1:
        parser.error("--step-min must be at least 1")
    key = None
    if not args.no_collect:
        key = ct.load_api_key()
        if not key or key == "your_key_here":
            say("TOMTOM_API_KEY is not set in .env")
            return 1

    if args.step_min != STEP_MINUTES:
        say("*** PIPELINE TEST MODE: readings every %d minutes. The model was trained on 15-minute "
            "readings, so this checks the plumbing only, not accuracy. ***" % args.step_min)

    say("Loading the model (about a minute)...")
    kwargs = {"forecast_dir": args.forecast_dir} if args.forecast_dir else {}
    engine = ForecastEngine(BangaloreForecaster(step_minutes=args.step_min), args.readings,
                            **kwargs)
    say("Model ready.")

    stop = threading.Event()
    kafka_stop = threading.Event()
    raw_producer = JsonProducer(args.kafka_bootstrap, RAW_TOPIC)
    prediction_producer = JsonProducer(args.kafka_bootstrap, PREDICTIONS_TOPIC)

    def process_raw_poll(poll_id, timestamp, readings, marker):
        event = engine.process_raw_batch(poll_id, timestamp, readings, marker)
        prediction_producer.send(event, key=poll_id)
        say("[kafka] published forecast for poll %s (%d%% sensor coverage, %d warm-up step(s))" % (
            poll_id, event["sensor_coverage_pct"], event["warmup_steps"]))
        return True

    assembler = RawPollAssembler(process_raw_poll)
    raw_consumer = JsonConsumer(args.kafka_bootstrap, RAW_TOPIC, "traffic-model", assembler.handle,
                                kafka_stop, auto_offset_reset="latest",
                                commit_when=lambda handled: bool(handled))
    prediction_consumer = JsonConsumer(
        args.kafka_bootstrap, PREDICTIONS_TOPIC, "traffic-dashboard", engine.accept_prediction,
        kafka_stop, auto_offset_reset="earliest")

    raw_consumer.start()
    prediction_consumer.start()

    import uvicorn
    app_server = uvicorn.Server(uvicorn.Config(create_app(engine=engine), host=args.host,
                                                port=args.port, log_level="warning"))
    api_thread = threading.Thread(target=app_server.run, name="traffic-http", daemon=True)
    api_thread.start()
    say("Forecast API: http://%s:%d/health (HTTP output is consumed from traffic.predictions)" % (
        args.host, args.port))

    collector_thread = None
    if not args.no_collect:
        junctions = ct.load_junctions()

        def publish_poll(stamp, rows):
            poll_id = stamp
            sent = raw_producer.send_poll(poll_id, stamp, rows, len(junctions),
                                          len(junctions) - len(rows))
            say("[kafka] published poll %s to %s (%d readings + completion marker)" % (
                poll_id, RAW_TOPIC, sent))

        collector = ct.Collector(
            ct.make_tomtom_fetch(key), junctions, out_path=args.readings,
            interval_min=args.step_min, sleep_fn=lambda seconds: stop.wait(seconds),
            write_csv=False, on_rows=publish_poll)

        def collect():
            try:
                collector.run(max_polls=args.max_polls, log=say,
                              should_stop=stop.is_set)
                say("[collector] stopped")
            except Exception as exc:
                LOG.exception("Collector stopped after an error")
                say("[collector] stopped: %s" % exc)

        say("Polling TomTom every %d minutes; raw readings go to %s." % (
            args.step_min, RAW_TOPIC))
        collector_thread = threading.Thread(target=collect, name="tomtom-collector", daemon=True)
        collector_thread.start()

    try:
        if args.interactive:
            terminal_loop(engine, stop)
        else:
            while not stop.is_set():
                stop.wait(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        kafka_stop.set()
        app_server.should_exit = True
        raw_consumer.close()
        prediction_consumer.close()
        if collector_thread is not None:
            collector_thread.join(timeout=5)
        raw_consumer.join(timeout=5)
        prediction_consumer.join(timeout=5)
        api_thread.join(timeout=5)
        raw_producer.close()
        prediction_producer.close()
    say("Stopped.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
