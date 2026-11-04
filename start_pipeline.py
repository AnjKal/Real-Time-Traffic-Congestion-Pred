"""One command that runs the whole live pipeline (Python 3.7 / TensorFlow env).

    python start_pipeline.py                  # PIPELINE TEST: readings every 2 minutes
    python start_pipeline.py --step-min 15    # real mode: readings every 15 minutes

What happens:
  1. The trained model loads (about a minute).
  2. A background collector polls TomTom for all 100 junctions on the clock (every 2 minutes in
     test mode) and appends rows, in the training table's layout, to data/live/readings.csv.
  3. After EVERY poll the forecast for all 100 junctions is refreshed (+2 ... +24 minutes in
     test mode). It needs 12 readings first, so the first forecast appears after the 12th poll.
  4. At the prompt, type a junction name or ID (e.g. "silk board" or J001) to see that
     junction's current forecast. Type "list" for all junctions, "quit" to stop.
  5. The same forecasts are served over HTTP on 127.0.0.1:8001 for the web page
     (/forecast, /node?name=..., /nodes, /health).

The collector stops by itself when the daily TomTom quota guard (default 2,400 calls = 24 polls)
is reached; the last forecast stays available.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import collect_tomtom as ct
from model_service import ForecastEngine, create_app
from predict_bangalore import BangaloreForecaster, STEP_MINUTES

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
            node["input_last_slot"][11:19], node["data_age_min"])]
    if node.get("warning"):
        lines.append("*** PIPELINE TEST: plumbing check, not a measure of model accuracy ***")
    for h in node["horizons"]:
        lines.append("  +%2d min (%s)   %5.1f km/h   %s" % (
            h["minutes_ahead"], h["forecast_for"][11:16], h["predicted_kmh"], h["band"]))
    return "\n".join(lines)


def progress_text(engine):
    h = engine.health()
    have = h.get("distinct_batches", 0)
    need = h["slots_needed"]
    if h["ready"]:
        return "forecast refreshed (newest reading %s)" % h["newest_slot"][11:19]
    left = max(0, need - have)
    return "waiting for readings: %d of %d, first forecast after %d more poll(s) (~%d min)" % (
        have, need, left, left * engine.step)


def terminal_loop(engine, stop):
    say("\nType a junction name or ID (e.g. silk board, J001), 'list', or 'quit'.")
    while not stop.is_set():
        try:
            line = input("> ").strip()
        except EOFError:                       # no keyboard attached: just keep running
            stop.wait()
            return
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            return
        if line.lower() == "list":
            for j in engine.list_junctions():
                say("  %s  %s" % (j["junction_id"], j["name"] or ""))
            continue
        try:
            say(format_node(engine.node_forecast(line)))
        except KeyError as e:
            say("  " + str(e.args[0]))
        except ValueError as e:
            say("  " + str(e))
        except LookupError:
            say("  No forecast yet: " + progress_text(engine))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-min", type=int, default=2,
                        help="minutes between readings: 2 = pipeline test (default), 15 = real")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--daily-quota", type=int, default=2400)
    parser.add_argument("--max-polls", type=int, default=None)
    parser.add_argument("--readings", default=ct.READINGS_PATH, help="readings CSV (read and written)")
    parser.add_argument("--forecast-dir", default=None, help="where forecasts are saved")
    parser.add_argument("--no-collect", action="store_true",
                        help="do not poll TomTom; serve forecasts from existing readings only")
    args = parser.parse_args()
    step = args.step_min

    key = None
    if not args.no_collect:
        key = ct.load_api_key()
        if not key or key == "your_key_here":
            say("TOMTOM_API_KEY is not set in .env")
            return 1
    if step != STEP_MINUTES:
        say("*** PIPELINE TEST MODE: readings every %d minutes. The model was trained on 15-minute "
            "readings, so this checks the plumbing only, not accuracy. ***" % step)

    say("Loading the model (about a minute)...")
    kwargs = {"forecast_dir": args.forecast_dir} if args.forecast_dir else {}
    engine = ForecastEngine(BangaloreForecaster(step_minutes=step), args.readings, **kwargs)
    say("Model ready.")

    import uvicorn
    server = uvicorn.Server(uvicorn.Config(create_app(engine=engine), host="127.0.0.1",
                                           port=args.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    say("Forecast API: http://127.0.0.1:%d/health  (web page reads this)" % args.port)

    stop = threading.Event()
    if not args.no_collect:
        collector = ct.Collector(ct.make_tomtom_fetch(key), ct.load_junctions(),
                                 out_path=args.readings, interval_min=step,
                                 daily_quota=args.daily_quota, sleep_fn=lambda s: stop.wait(s))

        def after_poll(summary):
            try:
                engine.current_window()                     # refresh forecasts for all 100 nodes
                say("[forecast] " + progress_text(engine))
            except LookupError:
                say("[forecast] " + progress_text(engine))
            except Exception as e:                          # noqa: BLE001 - never kill the poller
                say("[forecast] ERROR while refreshing: %r" % (e,))

        def collect():
            try:
                collector.run(max_polls=args.max_polls, log=say, on_poll=after_poll,
                              should_stop=stop.is_set)
                say("[collector] finished (quota or poll limit reached). The latest forecast "
                    "stays available.")
            except ct.FatalError as e:
                say("[collector] STOPPED: %s" % e)

        say("Polling TomTom every %d minutes (up to %d calls today). First poll at the next "
            "%d-minute mark." % (step, args.daily_quota, step))
        threading.Thread(target=collect, daemon=True).start()

    try:
        terminal_loop(engine, stop)
    except KeyboardInterrupt:
        pass
    stop.set()
    server.should_exit = True
    say("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
