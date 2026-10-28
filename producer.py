"""Replay Bangalore features into Kafka: one message per daily row (all 16 sensors).

The dataset is sampled daily. Replaying one row per 20 seconds changes wall-clock speed only;
the model still sees 12 consecutive daily steps, and every message carries the original
dataset timestamp (event time).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from kafka import KafkaProducer

import settings
from create_topics import ensure_topics
from lib.bangalore_data import read_feature_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=20.0, help="seconds between rows (default 20)")
    ap.add_argument("--start-row", type=int, default=None,
                    help="first row to replay (default: start of the test split = 80%% of the file)")
    ap.add_argument("--start-hour", type=int, default=None,
                    help="start at HH:00 of the first weekday in the test split (e.g. 16 = afternoon rush hour, 7 = morning)")
    ap.add_argument("--prefill", type=int, default=settings.SEQ_LEN,
                    help="send this many rows immediately so the first forecast appears at once (0 = off)")
    ap.add_argument("--loop", action="store_true", help="restart from --start-row when the data ends")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    tables = read_feature_tables(settings.H5_PATH)
    with open(settings.FEATURE_STATS) as f:
        import json
        feature_stats = json.load(f)
    channels = feature_stats["_channels"]
    speed = tables["speed"].reindex(columns=settings.load_sensor_ids())
    dates = speed.index
    values = np.zeros((len(speed), settings.NUM_NODES, settings.INPUT_DIM), dtype=np.float32)
    values[..., 0] = speed.fillna(0.0).values
    for channel_index, name in enumerate(channels[1:], start=1):
        key = name.replace("=", "_")
        if key in tables:
            feature = tables[key].reindex(index=dates, columns=settings.load_sensor_ids()).fillna(0.0)
            if name in feature_stats:
                feature = (feature - feature_stats[name]["mean"]) / feature_stats[name]["std"]
            values[..., channel_index] = feature.values
        elif name == "observed":
            values[..., channel_index] = speed.notna().values
        elif name == "day_of_week":
            values[..., channel_index] = np.repeat((dates.dayofweek.values / 7.0)[:, None], settings.NUM_NODES, axis=1)
        elif name == "is_weekend":
            values[..., channel_index] = np.repeat((dates.dayofweek.values >= 5).astype(float)[:, None], settings.NUM_NODES, axis=1)
        else:
            raise SystemExit("Missing feature channel '%s' in %s" % (name, settings.H5_PATH))
    ids = [str(c) for c in speed.columns]
    if os.path.exists(settings.SENSOR_IDS_TXT) and ids != settings.load_sensor_ids():
        print("WARNING: column order of %s differs from graph_sensor_ids.txt. "
              "The model's graph assumes the graph order!" % settings.H5_PATH)
    if len(ids) != settings.NUM_NODES:
        sys.exit("Expected %d sensors, found %d" % (settings.NUM_NODES, len(ids)))

    n = len(speed)
    start = args.start_row if args.start_row is not None else int(0.8 * n)  # test split, no train leakage
    if args.start_row is None and args.start_hour is not None:
        idx = dates
        ok = (np.arange(n) >= int(0.8 * n)) & (idx.hour == args.start_hour) & (idx.minute == 0) & (idx.dayofweek < 5)
        found = np.where(ok)[0]
        if len(found) == 0:
            sys.exit("No weekday row at %02d:00 in the test split" % args.start_hour)
        start = int(found[0])
    raw_speeds = np.nan_to_num(speed.values.astype(float))

    ensure_topics()
    producer = KafkaProducer(
        bootstrap_servers=settings.BOOTSTRAP,
        acks="all",          # wait for the broker to persist
        retries=5,
        key_serializer=lambda k: k.encode(),
        value_serializer=lambda v: json.dumps(v).encode(),
    )

    i, sent = start, 0
    print("Replaying %d Bangalore rows from row %d every %.1fs" % (n - start, start, args.interval))
    try:
        while True:
            if i >= n:
                if args.loop:
                    i = start
                else:
                    break
            if args.max_rows is not None and sent >= args.max_rows:
                break
            msg = {
                "row": int(i),
                "ts": dates[i].isoformat(),
                "speeds": np.round(raw_speeds[i], 2).tolist(),
                "features": values[i].tolist(),
            }
            meta = producer.send(settings.TOPIC_RAW, key="bangalore", value=msg).get(timeout=15)
            print("sent row %d ts=%s -> %s[%d]@%d" % (i, msg["ts"], meta.topic, meta.partition, meta.offset))
            i += 1
            sent += 1
            if sent >= args.prefill:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        producer.close()
        print("producer stopped after %d rows" % sent)


if __name__ == "__main__":
    main()