
import csv
import json
import time
import argparse
from kafka import KafkaProducer

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "traffic-data"
CSV_PATH = "data/BANGALORE/traffic_5mins/bengaluru_traffic_5min_synthetic.csv"

NUMERIC_COLUMNS = {
    "Traffic Volume",
    "Average Speed",
    "Travel Time Index",
    "Congestion Level",
    "Road Capacity Utilization",
    "Incident Reports",
    "Environmental Impact",
    "Public Transport Usage",
    "Traffic Signal Compliance",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "Hour",
    "Minute",
    "DayOfWeek",
    "IsWeekend",
    "IsHoliday",
    "Rainfall_mm",
    "Temperature_C",
    "Visibility_km",
    "IncidentSeverity"
}

def convert_record(row):
    record = {}

    for key, value in row.items():
        if key in NUMERIC_COLUMNS and value != "":
            number = float(value)
            record[key] = int(number) if number.is_integer() else number
        else:
            record[key] = value

    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5
    )

    print("Connected to Kafka")
    print(f"Publishing messages to topic: {TOPIC}")

    try:
        while True:
            count = 0

            with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    if args.limit is not None and count >= args.limit:
                        break
                    record = convert_record(row)

                    road = record["Road/Intersection Name"]

                    producer.send(
                        TOPIC,
                        key=road,
                        value=record
                    )

                    count += 1

                    if count % 100 == 0:
                        producer.flush()
                        print(
                            f"Published: {count} | "
                            f"Road: {road} | "
                            f"Speed: {record['Average Speed']} km/h"
                        )

                    time.sleep(args.delay)

            producer.flush()
            print(f"Finished publishing {count} traffic records")

            if not args.loop:
                break

            print("Restarting dataset stream...")

    except KeyboardInterrupt:
        print("Producer stopped by user")

    finally:
        producer.flush()
        producer.close()

if __name__ == "__main__":
    main()
