import json
from datetime import datetime, timedelta
from kafka import KafkaConsumer, KafkaProducer
from database import get_connection

BOOTSTRAP_SERVERS = "localhost:9092"
INPUT_TOPIC = "traffic-data"
OUTPUT_TOPIC = "traffic-aggregates"
GROUP_ID = "traffic-aggregation-group-v2"


def get_window_start(timestamp):
    dt = datetime.fromisoformat(timestamp)
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def create_aggregate():
    return {
        "record_count": 0,
        "total_traffic_volume": 0,
        "speed_sum": 0.0,
        "congestion_sum": 0.0,
        "max_capacity_utilization": 0.0,
        "incident_count": 0
    }


def add_record(aggregate, record):
    aggregate["record_count"] += 1
    aggregate["total_traffic_volume"] += float(record.get("Traffic Volume", 0))
    aggregate["speed_sum"] += float(record.get("Average Speed", 0))
    aggregate["congestion_sum"] += float(record.get("Congestion Level", 0))
    aggregate["max_capacity_utilization"] = max(
        aggregate["max_capacity_utilization"],
        float(record.get("Road Capacity Utilization", 0))
    )
    aggregate["incident_count"] += int(float(record.get("Incident Reports", 0)))


def save_to_database(result):
    connection = get_connection()

    if connection is None:
        raise RuntimeError("PostgreSQL connection unavailable")

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO traffic_readings (
                road_name,
                recorded_at,
                traffic_volume,
                average_speed,
                congestion_level,
                capacity_utilization,
                incident_count,
                window_minutes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (road_name, recorded_at, window_minutes)
            DO NOTHING
            """,
            (
                result["road"],
                datetime.fromisoformat(result["window_start"]),
                result["total_traffic_volume"],
                result["average_speed"],
                result["average_congestion"],
                result["max_capacity_utilization"],
                result["incident_count"],
                5
            )
        )

        connection.commit()
        cursor.close()

        print(
            f"Database saved: {result['road']} | "
            f"{result['window_start']}"
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def publish_window(producer, window_start, road, aggregate):
    count = aggregate["record_count"]

    if count == 0:
        return

    result = {
        "window_start": window_start.isoformat(),
        "window_end": (window_start + timedelta(minutes=5)).isoformat(),
        "road": road,
        "record_count": count,
        "total_traffic_volume": aggregate["total_traffic_volume"],
        "average_speed": round(aggregate["speed_sum"] / count, 2),
        "average_congestion": round(aggregate["congestion_sum"] / count, 2),
        "max_capacity_utilization": round(
            aggregate["max_capacity_utilization"], 2
        ),
        "incident_count": aggregate["incident_count"],
        "source": "synthetic_csv_replay"
    }

    producer.send(
        OUTPUT_TOPIC,
        key=road,
        value=result
    ).get(timeout=30)

    save_to_database(result)

    print(
        f"Aggregated: {road} | "
        f"Window: {window_start} | "
        f"Avg speed: {result['average_speed']} km/h | "
        f"Volume: {result['total_traffic_volume']}"
    )


def main():
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        key_deserializer=lambda key: key.decode("utf-8") if key else None,
        value_deserializer=lambda value: json.loads(value.decode("utf-8"))
    )

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        acks="all",
        retries=5,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8")
    )

    active_windows = {}
    latest_window_by_road = {}

    print("Kafka consumer connected")
    print(f"Reading from: {INPUT_TOPIC}")
    print(f"Publishing to: {OUTPUT_TOPIC}")
    print("PostgreSQL storage enabled")

    try:
        for message in consumer:
            record = message.value
            timestamp = record.get("Timestamp")
            road = record.get("Road/Intersection Name")

            if not timestamp or not road:
                continue

            window_start = get_window_start(timestamp)
            previous_window = latest_window_by_road.get(road)

            if previous_window is not None and window_start < previous_window:
                print(f"Out-of-order record for {road}: {timestamp}")
                continue

            if previous_window is not None and window_start > previous_window:
                old_key = (road, previous_window)
                old_aggregate = active_windows.pop(old_key)

                publish_window(
                    producer,
                    previous_window,
                    road,
                    old_aggregate
                )

                consumer.commit()

            latest_window_by_road[road] = window_start
            key = (road, window_start)

            if key not in active_windows:
                active_windows[key] = create_aggregate()

            add_record(active_windows[key], record)

    except KeyboardInterrupt:
        print("Stopping consumer...")

    finally:
        try:
            for (road, window_start), aggregate in sorted(
                active_windows.items(),
                key=lambda item: (item[0][1], item[0][0])
            ):
                publish_window(producer, window_start, road, aggregate)

            consumer.commit()

        finally:
            producer.flush()
            consumer.close()
            producer.close()


if __name__ == "__main__":
    main()