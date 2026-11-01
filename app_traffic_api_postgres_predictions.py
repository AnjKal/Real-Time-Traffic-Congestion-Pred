import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from torch import nn

from database import get_connection

MODEL_PATH = os.getenv("TRAFFIC_MODEL", "models/traffic_gru_best.pt")
CSV_PATH = os.getenv(
    "TRAFFIC_CSV",
    "data/BANGALORE/traffic_5mins/bengaluru_traffic_5min_synthetic.csv"
)


class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size=96, layers=2, dropout=0.15, horizon=12):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}. Train the GRU first.")

try:
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roads = ckpt["roads"]
numeric_features = ckpt["numeric_features"]
weather_categories = ckpt["weather_categories"]
seq_len = int(ckpt["seq_len"])
horizon = int(ckpt["horizon"])

model = TrafficGRU(
    int(ckpt["input_size"]),
    int(ckpt["hidden_size"]),
    int(ckpt["layers"]),
    float(ckpt["dropout"]),
    horizon
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

app = Flask(__name__)
CORS(app)


def db_query(query, params=None, fetch="all"):
    connection = get_connection()
    if connection is None:
        raise ConnectionError("Could not connect to PostgreSQL.")
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            if fetch == "one":
                result = cursor.fetchone()
            elif fetch == "none":
                result = None
            else:
                result = cursor.fetchall()
        connection.commit()
        return result
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def read_traffic_csv():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Traffic CSV not found: {CSV_PATH}. Set TRAFFIC_CSV or place the generated CSV at this path."
        )
    frame = pd.read_csv(CSV_PATH)
    if "Timestamp" not in frame.columns:
        raise ValueError("The traffic CSV must contain a Timestamp column.")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    return frame.dropna(subset=["Timestamp"])


def prepare_observations(records, road):
    frame = pd.DataFrame(records)
    required = numeric_features + ["Weather Conditions", "Timestamp"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing observation fields: " + ", ".join(missing))
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    if frame["Timestamp"].isna().any():
        raise ValueError("Invalid or missing Timestamp.")
    frame = frame.sort_values("Timestamp").reset_index(drop=True)
    if len(frame) != seq_len:
        raise ValueError(f"Exactly {seq_len} observations are required.")
    if not (frame["Timestamp"].diff().dropna() == pd.Timedelta(minutes=5)).all():
        raise ValueError("The 12 observations must be exactly 5 minutes apart.")

    numeric = frame[numeric_features].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    weather = pd.get_dummies(frame["Weather Conditions"].fillna("Unknown"))
    weather = weather.reindex(columns=weather_categories, fill_value=0)
    road_features = np.zeros((len(frame), len(roads)), dtype=np.float32)
    road_features[:, roads.index(road)] = 1.0
    features = np.concatenate([
        numeric.to_numpy(dtype=np.float32),
        weather.to_numpy(dtype=np.float32),
        road_features
    ], axis=1)
    if features.shape[1] != int(ckpt["input_size"]):
        raise ValueError("Feature count does not match the checkpoint.")
    mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    std = np.asarray(ckpt["x_std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    features = (features - mean) / std
    return torch.tensor(features[None], dtype=torch.float32), frame["Timestamp"].iloc[-1]


def infer(records, road):
    x, last_time = prepare_observations(records, road)
    with torch.no_grad():
        scaled = model(x.to(device)).cpu().numpy()[0]
    speeds = np.clip(scaled * float(ckpt["y_std"]) + float(ckpt["y_mean"]), 0, 150)
    predictions = []
    for index, speed in enumerate(speeds, start=1):
        future_time = last_time + pd.Timedelta(minutes=5 * index)
        predictions.append({
            "horizon_minutes": 5 * index,
            "forecast_timestamp": future_time.isoformat(),
            "predicted_average_speed_kmh": round(float(speed), 2)
        })
    return predictions, last_time



def save_predictions(road, predictions):
    connection = get_connection()
    if connection is None:
        raise ConnectionError("Could not connect to PostgreSQL.")

    rows = []
    for item in predictions:
        forecast_time = pd.to_datetime(item["forecast_timestamp"]).to_pydatetime()
        rows.append((
            road,
            forecast_time,
            int(item["horizon_minutes"]),
            float(item["predicted_average_speed_kmh"]),
            "GRU"
        ))

    try:
        with connection.cursor() as cursor:
            if rows:
                start_time = min(row[1] for row in rows)
                end_time = max(row[1] for row in rows)
                cursor.execute(
                    """
                    DELETE FROM traffic_predictions
                    WHERE road_name = %s
                      AND prediction_time BETWEEN %s AND %s
                      AND model_name = %s
                    """,
                    (road, start_time, end_time, "GRU")
                )
                cursor.executemany(
                    """
                    INSERT INTO traffic_predictions
                        (road_name, prediction_time, forecast_minutes,
                         predicted_speed, model_name)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    rows
                )
        connection.commit()
        return len(rows)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def classify_speed(speed):
    if speed is None or pd.isna(speed):
        return "unknown"
    if speed >= 40:
        return "free"
    if speed >= 25:
        return "moderate"
    return "heavy"


def iso_value(value):
    return value.isoformat() if value is not None else None


def get_active_issues():
    rows = db_query(
        """
        SELECT id, road_name, incident_type, severity, description, reported_at, status
        FROM road_incidents
        WHERE status <> 'Resolved'
        ORDER BY reported_at DESC
        """
    )
    return [dict(row) for row in rows]


def road_rows():
    traffic = db_query(
        """
        SELECT DISTINCT ON (road_name)
            road_name, recorded_at, traffic_volume, average_speed,
            congestion_level, capacity_utilization, incident_count
        FROM traffic_readings
        ORDER BY road_name, recorded_at DESC, id DESC
        """
    )
    issues = get_active_issues()
    traffic_by_road = {row["road_name"]: row for row in traffic}
    rows = []
    for road in roads:
        latest = traffic_by_road.get(road)
        if latest is None:
            continue
        active = [item for item in issues if item["road_name"] == road]
        speed = float(latest["average_speed"]) if latest["average_speed"] is not None else None
        status = classify_speed(speed)
        if any(item["severity"] == "High" for item in active) and status == "free":
            status = "moderate"
        rows.append({
            "road": road,
            "area": "Bengaluru",
            "latest_speed_kmh": round(speed, 2) if speed is not None else None,
            "traffic_volume": int(latest["traffic_volume"]) if latest["traffic_volume"] is not None else None,
            "congestion_level": float(latest["congestion_level"]) if latest["congestion_level"] is not None else None,
            "last_observed_timestamp": iso_value(latest["recorded_at"]),
            "status": status,
            "active_issue_count": len(active),
            "issue_types": [item["incident_type"] for item in active]
        })
    return rows


def serialize_issue(row):
    return {
        "id": str(row["id"]),
        "road": row["road_name"],
        "type": row["incident_type"],
        "severity": row["severity"],
        "description": row["description"],
        "source": "Dashboard report",
        "status": row["status"],
        "created_at": iso_value(row["reported_at"])
    }


@app.get("/")
def root():
    return jsonify({
        "message": "Bengaluru traffic prediction API",
        "endpoints": ["/api/health", "/api/roads", "/api/predict", "/api/dashboard",
                      "/api/predict/latest", "/api/predictions/latest", "/api/road-issues"]
    })


@app.get("/api/health")
def health():
    try:
        db_query("SELECT 1", fetch="one")
        database_status = "connected"
    except Exception as exc:
        database_status = "error"
        return jsonify({
            "status": "degraded",
            "model": "GRU",
            "device": str(device),
            "database": database_status,
            "database_error": str(exc),
            "data_type": ckpt.get("data_type", "UNKNOWN")
        }), 503
    return jsonify({
        "status": "ok",
        "model": "GRU",
        "device": str(device),
        "database": database_status,
        "data_type": ckpt.get("data_type", "UNKNOWN")
    })


@app.get("/api/roads")
def get_roads():
    return jsonify({"roads": roads})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    road = payload.get("road")
    observations = payload.get("observations")
    if road not in roads:
        return jsonify({"error": "Unknown road.", "available_roads": roads}), 400
    if not isinstance(observations, list):
        return jsonify({"error": "observations must be an array of 12 records."}), 400
    try:
        predictions, last_time = infer(observations, road)
        saved_count = save_predictions(road, predictions)
        return jsonify({
            "road": road,
            "last_observed_timestamp": last_time.isoformat(),
            "data_type": ckpt.get("data_type", "UNKNOWN"),
            "saved_to_database": saved_count,
            "predictions": predictions,
            "warning": "Synthetic-data prototype; not validated real-world traffic predictions."
                       if ckpt.get("data_type") == "SYNTHETIC" else None
        })
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/predict/latest")
def predict_latest():
    road = request.args.get("road")
    if road not in roads:
        return jsonify({"error": "Unknown or missing road.", "available_roads": roads}), 400
    try:
        frame = read_traffic_csv()
        part = frame[frame["Road/Intersection Name"] == road].sort_values("Timestamp").tail(seq_len)
        if len(part) != seq_len:
            return jsonify({"error": f"Not enough observations for {road}."}), 400
        records = json.loads(part.to_json(orient="records", date_format="iso"))
        predictions, last_time = infer(records, road)
        saved_count = save_predictions(road, predictions)
        return jsonify({
            "road": road,
            "last_observed_timestamp": last_time.isoformat(),
            "data_type": ckpt.get("data_type", "UNKNOWN"),
            "saved_to_database": saved_count,
            "predictions": predictions,
            "warning": "Synthetic-data prototype; not validated real-world traffic predictions."
                       if ckpt.get("data_type") == "SYNTHETIC" else None
        })
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400




@app.get("/api/predictions/latest")
def get_latest_saved_predictions():
    road = request.args.get("road")
    if road and road not in roads:
        return jsonify({"error": "Unknown road.", "available_roads": roads}), 400

    try:
        if road:
            rows = db_query(
                """
                SELECT id, road_name, prediction_time, forecast_minutes,
                       predicted_speed, predicted_congestion, model_name, created_at
                FROM traffic_predictions
                WHERE road_name = %s
                ORDER BY created_at DESC, prediction_time ASC
                LIMIT %s
                """,
                (road, horizon)
            )
        else:
            rows = db_query(
                """
                SELECT DISTINCT ON (road_name, forecast_minutes)
                       id, road_name, prediction_time, forecast_minutes,
                       predicted_speed, predicted_congestion, model_name, created_at
                FROM traffic_predictions
                ORDER BY road_name, forecast_minutes, created_at DESC
                """
            )

        return jsonify({
            "predictions": [
                {
                    "id": row["id"],
                    "road": row["road_name"],
                    "forecast_timestamp": iso_value(row["prediction_time"]),
                    "horizon_minutes": row["forecast_minutes"],
                    "predicted_average_speed_kmh": (
                        round(float(row["predicted_speed"]), 2)
                        if row["predicted_speed"] is not None else None
                    ),
                    "predicted_congestion": row["predicted_congestion"],
                    "model_name": row["model_name"],
                    "created_at": iso_value(row["created_at"])
                }
                for row in rows
            ]
        })
    except Exception as exc:
        app.logger.exception("Could not load saved predictions")
        return jsonify({"error": str(exc)}), 503


@app.get("/api/dashboard")
def dashboard():
    try:
        rows = road_rows()
        counts = {
            "total_roads": len(rows),
            "free_roads": sum(row["status"] == "free" for row in rows),
            "moderate_roads": sum(row["status"] == "moderate" for row in rows),
            "heavy_roads": sum(row["status"] == "heavy" for row in rows)
        }
        return jsonify({
            "summary": counts,
            "roads": rows,
            "data_type": ckpt.get("data_type", "UNKNOWN"),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "source_timestamp": max((row["last_observed_timestamp"] for row in rows), default=None)
        })
    except Exception as exc:
        app.logger.exception("Dashboard query failed")
        return jsonify({"error": str(exc)}), 503


@app.get("/api/road-issues")
def list_issues():
    try:
        rows = db_query(
            """
            SELECT id, road_name, incident_type, severity, description, reported_at, status
            FROM road_incidents
            ORDER BY reported_at DESC
            """
        )
        return jsonify({"issues": [serialize_issue(row) for row in rows]})
    except Exception as exc:
        app.logger.exception("Could not load road incidents")
        return jsonify({"error": str(exc)}), 503


@app.post("/api/road-issues")
def create_issue():
    payload = request.get_json(silent=True) or {}
    road = payload.get("road")
    issue_type = payload.get("type")
    severity = payload.get("severity", "Moderate")
    description = str(payload.get("description", "")).strip()
    allowed_types = {"Construction", "Waterlogging", "Accident", "Road closure", "Pothole", "Signal failure"}
    if road not in roads:
        return jsonify({"error": "Select a valid road."}), 400
    if issue_type not in allowed_types:
        return jsonify({"error": "Select a valid issue type."}), 400
    if severity not in {"Low", "Moderate", "High"}:
        return jsonify({"error": "Select a valid severity."}), 400
    if not description:
        return jsonify({"error": "Please enter a short description."}), 400
    try:
        row = db_query(
            """
            INSERT INTO road_incidents
                (road_name, incident_type, severity, description, status)
            VALUES (%s, %s, %s, %s, 'Open')
            RETURNING id, road_name, incident_type, severity, description, reported_at, status
            """,
            (road, issue_type, severity, description),
            fetch="one"
        )
        return jsonify({"message": "Issue recorded.", "issue": serialize_issue(row)}), 201
    except Exception as exc:
        app.logger.exception("Could not create road incident")
        return jsonify({"error": str(exc)}), 503


@app.patch("/api/road-issues/<int:issue_id>")
def update_issue(issue_id):
    payload = request.get_json(silent=True) or {}
    status = payload.get("status")
    if status not in {"Open", "Investigating", "Resolved"}:
        return jsonify({"error": "Invalid issue status."}), 400
    try:
        row = db_query(
            """
            UPDATE road_incidents
            SET status = %s
            WHERE id = %s
            RETURNING id, road_name, incident_type, severity, description, reported_at, status
            """,
            (status, issue_id),
            fetch="one"
        )
        if row is None:
            return jsonify({"error": "Issue not found."}), 404
        return jsonify({"message": "Issue updated.", "issue": serialize_issue(row)})
    except Exception as exc:
        app.logger.exception("Could not update road incident")
        return jsonify({"error": str(exc)}), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
