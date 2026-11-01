import json
import signal
from collections import defaultdict
from datetime import datetime, timedelta

from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import OffsetAndMetadata, TopicPartition
from psycopg2.extras import execute_values

from database import get_connection

BOOTSTRAP_SERVERS = "localhost:9092"
INPUT_TOPIC = "traffic-data"
OUTPUT_TOPIC = "traffic-aggregates"
GROUP_ID = "traffic-aggregation-group-v2"
POLL_TIMEOUT_MS = 1000
MAX_POLL_RECORDS = 500
DB_BATCH_SIZE = 100

running = True


def stop_handler(signum, frame):
    global running
    running = False


def get_window_start(timestamp):
    dt = datetime.fromisoformat(timestamp)
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def create_aggregate(partition, first_offset):
    return {
        "record_count": 0,
        "total_traffic_volume": 0.0,
        "speed_sum": 0.0,
        "congestion_sum": 0.0,
        "max_capacity_utilization": 0.0,
        "incident_count": 0,
        "partition": partition,
        "first_offset": first_offset
    }


def add_record(aggregate, record):
    aggregate["record_count"] += 1
    aggregate["total_traffic_volume"] += float(record.get("Traffic Volume", 0) or 0)
    aggregate["speed_sum"] += float(record.get("Average Speed", 0) or 0)
    aggregate["congestion_sum"] += float(record.get("Congestion Level", 0) or 0)
    aggregate["max_capacity_utilization"] = max(
        aggregate["max_capacity_utilization"],
        float(record.get("Road Capacity Utilization", 0) or 0)
    )
    aggregate["incident_count"] += int(float(record.get("Incident Reports", 0) or 0))


def build_result(road, window_start, aggregate):
    count = aggregate["record_count"]
    if count == 0:
        return None

    return {
        "window_start": window_start.isoformat(),
        "window_end": (window_start + timedelta(minutes=5)).isoformat(),
        "road": road,
        "record_count": count,
        "total_traffic_volume": round(aggregate["total_traffic_volume"], 2),
        "average_speed": round(aggregate["speed_sum"] / count, 2),
        "average_congestion": round(aggregate["congestion_sum"] / count, 2),
        "max_capacity_utilization": round(aggregate["max_capacity_utilization"], 2),
        "incident_count": aggregate["incident_count"],
        "source": "synthetic_csv_replay"
    }


def save_batch_to_database(results):
    if not results:
        return

    rows = [
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
        for result in results
    ]

    connection = get_connection()
    if connection is None:
        raise RuntimeError("PostgreSQL connection unavailable")

    try:
        with connection.cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO traffic_readings (
                    road_name, recorded_at, traffic_volume, average_speed,
                    congestion_level, capacity_utilization, incident_count,
                    window_minutes
                )
                VALUES %s
                ON CONFLICT (road_name, recorded_at, window_minutes) DO NOTHING
                """,
                rows,
                page_size=DB_BATCH_SIZE
            )
        connection.commit()
        print(f"PostgreSQL batch saved: {len(results)} aggregate(s)")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def publish_results(producer, results):
    for result in results:
        producer.send(OUTPUT_TOPIC, key=result["road"], value=result)
    producer.flush(timeout=30)


def commit_safe_offsets(consumer, active_windows, last_processed):
    active_first_offsets = defaultdict(list)

    for aggregate in active_windows.values():
        active_first_offsets[aggregate["partition"]].append(aggregate["first_offset"])

    offsets = {}
    for partition, last_offset in last_processed.items():
        if active_first_offsets[partition]:
            safe_offset = min(active_first_offsets[partition])
        else:
            safe_offset = last_offset + 1

        if safe_offset > 0:
            offsets[TopicPartition(INPUT_TOPIC, partition)] = OffsetAndMetadata(safe_offset, "")

    if offsets:
        consumer.commit(offsets=offsets)


def finalize_windows(producer, active_windows, keys=None):
    selected_keys = list(active_windows.keys()) if keys is None else list(keys)
    results = []

    for key in selected_keys:
        aggregate = active_windows.pop(key, None)
        if aggregate is not None:
            result = build_result(key[0], key[1], aggregate)
            if result is not None:
                results.append(result)

    save_batch_to_database(results)
    publish_results(producer, results)

    for result in results:
        print(
            f"Aggregated: {result['road']} | Window: {result['window_start']} | "
            f"Avg speed: {result['average_speed']} km/h | "
            f"Volume: {result['total_traffic_volume']}"
        )


def main():
    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)

    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_records=MAX_POLL_RECORDS,
        key_deserializer=lambda key: key.decode("utf-8") if key else None,
        value_deserializer=lambda value: json.loads(value.decode("utf-8"))
    )

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        acks="all",
        retries=10,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8")
    )

    active_windows = {}
    latest_window_by_road = {}
    last_processed = {}

    print("Improved Kafka consumer connected")
    print(f"Reading from: {INPUT_TOPIC}")
    print(f"Publishing to: {OUTPUT_TOPIC}")
    print("PostgreSQL batch storage enabled")
    print("Manual partition-aware offset commits enabled")

    try:
        while running:
            batches = consumer.poll(timeout_ms=POLL_TIMEOUT_MS, max_records=MAX_POLL_RECORDS)
            closed_keys = []

            for topic_partition, messages in batches.items():
                for message in messages:
                    last_processed[message.partition] = message.offset
                    record = message.value
                    timestamp = record.get("Timestamp")
                    road = record.get("Road/Intersection Name")

                    if not timestamp or not road:
                        continue

                    try:
                        window_start = get_window_start(timestamp)
                    except (TypeError, ValueError):
                        print(f"Skipping invalid timestamp at offset {message.offset}")
                        continue

                    previous_window = latest_window_by_road.get(road)

                    if previous_window is not None and window_start < previous_window:
                        print(f"Out-of-order record skipped: {road} | {timestamp}")
                        continue

                    if previous_window is not None and window_start > previous_window:
                        old_key = (road, previous_window)
                        if old_key in active_windows:
                            closed_keys.append(old_key)

                    key = (road, window_start)
                    if key not in active_windows:
                        active_windows[key] = create_aggregate(message.partition, message.offset)

                    add_record(active_windows[key], record)
                    latest_window_by_road[road] = window_start

            if closed_keys:
                finalize_windows(producer, active_windows, list(dict.fromkeys(closed_keys)))

            commit_safe_offsets(consumer, active_windows, last_processed)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    finally:
        print("Flushing remaining aggregation windows...")
        try:
            finalize_windows(producer, active_windows)
            producer.flush(timeout=30)
            commit_safe_offsets(consumer, active_windows, last_processed)
        except Exception as error:
            print(f"Shutdown flush failed; uncommitted messages will be replayed: {error}")
        finally:
            consumer.close()
            producer.close()
            print("Consumer stopped")


if __name__ == "__main__":
    main()
