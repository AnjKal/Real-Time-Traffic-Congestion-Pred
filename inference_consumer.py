"""Kafka consumer that runs the trained Bangalore DCRNN on a sliding window and publishes forecasts.

traffic.raw (1 message = 16 speeds plus 19 features @ daily step)
    -> sliding window of 12 steps
    -> DCRNN (loaded exactly like run_demo.py: DCRNNSupervisor + supervisor.load)
    -> 12-step forecast (12..144 days ahead) for all 16 sensors
    -> congestion ratio / score / level per sensor and horizon
    -> traffic.predictions

Fault tolerance: offsets are committed manually, AFTER the forecast was published (at-least-once).
After a crash/restart the consumer rewinds SEQ_LEN-1 messages before the committed offset to
rebuild its sliding window, but does not re-publish forecasts it already produced.

Needs the TensorFlow 1.x environment of the DCRNN repo. Run from the repo root.
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.getcwd())  # DCRNN repo root: lib/, model/

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.structs import OffsetAndMetadata

import settings
from create_topics import ensure_topics
from lib.utils import load_graph_data
from lib.bangalore_data import read_feature_tables
from model.dcrnn_supervisor import DCRNNSupervisor


def offset_meta(offset):
    """OffsetAndMetadata signature differs between kafka-python versions (newer ones need leader_epoch)."""
    try:
        return OffsetAndMetadata(offset, "", -1)
    except TypeError:
        return OffsetAndMetadata(offset, "")


# ----------------------------------------------------------------------------- model
def resolve_model_filename(config):
    """Use the configured checkpoint or select the newest checkpoint from base_dir."""
    train_config = config.setdefault("train", {})
    model_filename = train_config.get("model_filename")
    if model_filename:
        path = Path(model_filename)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists() or Path(str(path) + ".index").exists():
            return str(path)
        raise SystemExit("Configured model checkpoint does not exist: %s" % path)

    base_dir = Path(config.get("base_dir", ""))
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    checkpoints = list(base_dir.rglob("*.index"))
    if not checkpoints:
        raise SystemExit(
            "No model checkpoint found under %s. Train the model first or pass "
            "a generated config_<epoch>.yaml." % base_dir
        )
    checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
    model_filename = str(checkpoint)[:-len(".index")]
    train_config["model_filename"] = model_filename
    print("No train.model_filename in config; using newest checkpoint: %s" % model_filename)
    return model_filename


def load_model(config_filename, cpu_only):
    """Same steps as run_demo.py, but keep the session open so we can feed live windows."""
    with open(config_filename) as f:
        config = yaml.safe_load(f)
    model_filename = resolve_model_filename(config)
    tf_config = tf.ConfigProto(device_count={"GPU": 0}) if cpu_only else tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    _, _, adj_mx = load_graph_data(config["data"]["graph_pkl_filename"])
    sess = tf.Session(config=tf_config)
    supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
    supervisor.load(sess, model_filename)
    return sess, supervisor


def get_scaler(supervisor):
    """The StandardScaler fitted on the training split (same one evaluate() uses)."""
    try:
        return supervisor._data["scaler"]
    except Exception:  # fallback: rebuild it from train.npz
        from lib.utils import StandardScaler
        train = np.load(os.path.join(settings.DATASET_DIR, "train.npz"))["x"]
        return StandardScaler(mean=train[..., 0].mean(), std=train[..., 0].std())


def predict(sess, model, scaler, batch_size, window):
    """window rows contain exactly SEQ_LEN daily observations and 19 features."""
    speeds = np.stack([s for _, s, _ in window])
    x = np.stack([features for _, _, features in window]).astype(np.float32)
    x[..., 0] = scaler.transform(speeds)
    feed = {model.inputs: np.repeat(x[None], batch_size, axis=0)}     # placeholder has a fixed batch size
    try:
        feed[model.labels] = np.zeros(model.labels.get_shape().as_list(), dtype=np.float32)
    except Exception:
        pass
    out = sess.run(model.outputs, feed_dict=feed)
    out = np.asarray(out).reshape(batch_size, settings.HORIZON, settings.NUM_NODES)[0]
    return np.clip(scaler.inverse_transform(out), 0.0, None)


# ------------------------------------------------------------------------ congestion
def free_flow_speeds():
    df = read_feature_tables(settings.H5_PATH)["speed"]
    train = df.values[: int(settings.TRAIN_FRACTION * len(df))].astype(float)
    train[~np.isfinite(train) | (train <= 0)] = np.nan  # zeros and invalid readings are missing
    ff = np.nanpercentile(train, settings.FREEFLOW_PERCENTILE, axis=0)
    ff = np.where(np.isnan(ff), 60.0, ff)   # sensors with no usable history
    return np.maximum(ff, settings.FREEFLOW_FLOOR_MPH)


def congestion(pred, ff):
    """pred: (HORIZON, 16) speed values. Returns score and congestion levels."""
    ratio = pred / ff[None, :]
    score = np.clip(1.0 - ratio, 0.0, 1.0) * 100.0
    t_free, t_mod = settings.LEVEL_THRESHOLDS
    level = np.select([ratio >= t_free, ratio >= t_mod], [0, 1], default=2)
    return score, level


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_filename", default=settings.MODEL_CONFIG)
    ap.add_argument("--use_cpu_only", action="store_true")
    ap.add_argument("--from-latest", action="store_true",
                    help="ignore committed offsets and old messages; start from the end of the topic")
    args = ap.parse_args()

    ensure_topics()
    ff = free_flow_speeds()

    print("Loading pretrained DCRNN ...")
    sess, supervisor = load_model(args.config_filename, args.use_cpu_only)
    model = supervisor._test_model
    scaler = get_scaler(supervisor)
    batch_size = model.inputs.get_shape().as_list()[0]
    print("Model ready (placeholder batch size %d)." % batch_size)

    tp = TopicPartition(settings.TOPIC_RAW, 0)
    consumer = KafkaConsumer(
        bootstrap_servers=settings.BOOTSTRAP,
        group_id=settings.GROUP_ID,
        enable_auto_commit=False,                 # we commit only after publishing the forecast
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode()),
    )
    consumer.assign([tp])
    producer = KafkaProducer(
        bootstrap_servers=settings.BOOTSTRAP, acks="all", retries=5,
        key_serializer=lambda k: k.encode(),
        value_serializer=lambda v: json.dumps(v).encode(),
    )

    begin = consumer.beginning_offsets([tp])[tp]
    if args.from_latest:
        resume = consumer.end_offsets([tp])[tp]
    else:
        committed = consumer.committed(tp)
        resume = committed if committed is not None else begin
    # rewind to rebuild the sliding window after a restart (rows before `resume` only warm the window)
    consumer.seek(tp, max(begin, resume - (settings.SEQ_LEN - 1)))
    print("Consuming %s from offset %d (publishing from offset %d)" % (settings.TOPIC_RAW, max(begin, resume - (settings.SEQ_LEN - 1)), resume))

    window = deque(maxlen=settings.SEQ_LEN)
    step = timedelta(minutes=settings.STEP_MIN)

    while True:
        records = consumer.poll(timeout_ms=1000, max_records=50).get(tp, [])
        for rec in records:
            msg = rec.value
            speeds = np.asarray(msg.get("speeds", []), dtype=float)
            if speeds.shape != (settings.NUM_NODES,):
                print("skipping malformed message at offset %d" % rec.offset)
                consumer.commit({tp: offset_meta(rec.offset + 1)})
                continue
            ts = pd.Timestamp(msg["ts"]).to_pydatetime()

            # a time jump (producer looped / rows missing) invalidates the window
            if window and (ts - window[-1][0]) != step:
                print("time gap at offset %d (%s -> %s): resetting window" % (rec.offset, window[-1][0], ts))
                window.clear()
            features = np.asarray(msg.get("features", []), dtype=np.float32)
            if features.shape != (settings.NUM_NODES, settings.INPUT_DIM):
                print("skipping message without %d feature channels at offset %d" % (settings.INPUT_DIM, rec.offset))
                consumer.commit({tp: offset_meta(rec.offset + 1)})
                continue
            window.append((ts, speeds, features))

            if rec.offset < resume:   # warm-up only, forecast for this row was already published
                continue
            if len(window) < settings.SEQ_LEN:
                print("warming up window %d/%d" % (len(window), settings.SEQ_LEN))
                consumer.commit({tp: offset_meta(rec.offset + 1)})
                continue

            t0 = time.time()
            pred = predict(sess, model, scaler, batch_size, list(window))   # (12, 207)
            score, level = congestion(pred, ff)
            infer_ms = (time.time() - t0) * 1000.0

            out = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data_ts": ts.isoformat(),                      # newest input row (dataset time)
                "offset": int(rec.offset),
                "horizon_minutes": [settings.STEP_MIN * (h + 1) for h in range(settings.HORIZON)],
                "target_ts": [(ts + step * (h + 1)).isoformat() for h in range(settings.HORIZON)],
                "current_speed": np.round(speeds, 1).tolist(),  # 0 = missing reading
                "pred_speed": np.round(pred, 1).tolist(),       # [12][207] mph
                "score": np.round(score, 1).tolist(),           # [12][207] 0..100
                "level": level.astype(int).tolist(),           # [12][207] 0..3
                "free_flow": np.round(ff, 1).tolist(),
                "inference_ms": round(infer_ms, 1),
            }
            producer.send(settings.TOPIC_PRED, key="bangalore", value=out).get(timeout=15)
            consumer.commit({tp: offset_meta(rec.offset + 1)})   # commit AFTER publish
            seen = speeds[speeds > 0]
            counts = [int((level[5] == k).sum()) for k in range(3)]
            print("offset %d  data_ts=%s  %.0f ms | mean speed now %.1f mph, +30min %.1f mph | "
                  "+30min free/moderate/congested = %d/%d/%d" %
                  (rec.offset, ts, infer_ms, seen.mean() if seen.size else 0.0, pred[5].mean(),
                   counts[0], counts[1], counts[2]))


if __name__ == "__main__":
    main()