import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "Traffic Volume", "Average Speed", "Travel Time Index",
    "Congestion Level", "Road Capacity Utilization", "Incident Reports",
    "Environmental Impact", "Public Transport Usage",
    "Traffic Signal Compliance", "Parking Usage",
    "Pedestrian and Cyclist Count"
]

def main():
    parser = argparse.ArgumentParser(
        description="Generate a clearly labelled synthetic 5-minute Bengaluru traffic dataset."
    )
    parser.add_argument(
        "--source",
        default="../Banglore_traffic_Dataset.csv",
        help="Path to the original daily CSV"
    )
    parser.add_argument(
        "--output",
        default="data/BANGALORE/traffic_5mins/bengaluru_traffic_5min_synthetic.csv"
    )
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Original daily dataset not found: {source_path.resolve()}. "
            "Pass its location using --source."
        )

    source = pd.read_csv(source_path)
    required = ["Date", "Area Name", "Road/Intersection Name",
                "Weather Conditions", "Roadwork and Construction Activity"] + NUMERIC_COLUMNS
    missing = [c for c in required if c not in source.columns]
    if missing:
        raise ValueError("Source CSV is missing columns: " + ", ".join(missing))

    source["Date"] = pd.to_datetime(source["Date"], errors="coerce")
    source = source.dropna(subset=["Date", "Road/Intersection Name"])
    source = source.sort_values(["Date", "Road/Intersection Name"])

    end_date = source["Date"].max().normalize()
    start_date = end_date - pd.Timedelta(days=args.days - 1)
    dates = pd.date_range(start_date, end_date, freq="D")

    roads = sorted(source["Road/Intersection Name"].astype(str).unique())
    area_by_road = source.groupby("Road/Intersection Name")["Area Name"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    ).to_dict()
    stats = source.groupby("Road/Intersection Name")[NUMERIC_COLUMNS].median()
    global_stats = source[NUMERIC_COLUMNS].median(numeric_only=True)
    weather_modes = source.groupby("Road/Intersection Name")["Weather Conditions"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else "Clear"
    ).to_dict()
    roadwork_modes = source.groupby("Road/Intersection Name")["Roadwork and Construction Activity"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else "No"
    ).to_dict()

    rng = np.random.default_rng(args.seed)
    records = []

    for day in dates:
        daily = source[source["Date"].dt.normalize() == day]
        dow = day.dayofweek
        weekend = dow >= 5
        holiday = (day.month, day.day) in {
            (1, 1), (8, 15), (10, 2), (12, 25)
        }

        for road in roads:
            observed = daily[daily["Road/Intersection Name"].astype(str) == road]
            anchor = observed.iloc[0] if not observed.empty else None
            road_stats = stats.loc[road] if road in stats.index else global_stats
            values = {
                c: float(anchor[c]) if anchor is not None and pd.notna(anchor[c])
                else float(road_stats[c])
                for c in NUMERIC_COLUMNS
            }
            weather = str(anchor["Weather Conditions"]) if anchor is not None else str(weather_modes.get(road, "Clear"))
            roadwork = str(anchor["Roadwork and Construction Activity"]) if anchor is not None else str(roadwork_modes.get(road, "No"))

            for step in range(288):
                timestamp = day + pd.Timedelta(minutes=5 * step)
                hour = timestamp.hour + timestamp.minute / 60
                morning = np.exp(-0.5 * ((hour - 9.0) / 1.45) ** 2)
                evening = np.exp(-0.5 * ((hour - 18.2) / 1.65) ** 2)
                peak = max(morning, evening)
                night_factor = 0.22 if hour < 6 else 1.0
                profile = (0.28 + 0.72 * peak) * night_factor
                if weekend:
                    profile *= 0.78
                if holiday:
                    profile *= 0.65

                phase = 0.04 * np.sin(2 * np.pi * step / 288 + roads.index(road) * 0.31)
                factor = max(0.12, profile * (1 + phase) * rng.normal(1, 0.035))
                volume = max(0, int(values["Traffic Volume"] * factor))
                speed = np.clip(
                    values["Average Speed"] + 13 * (1 - factor) - 8 * peak + rng.normal(0, 2),
                    5, 85
                )

                rainy = weather.lower() in {"rain", "rainy"}
                rainfall = max(0, rng.gamma(1.4, 1.1)) if rainy else max(0, rng.normal(0, 0.02))
                temperature = np.clip(
                    27 + 3 * np.sin(2 * np.pi * (hour - 14) / 24)
                    - (2 if rainy else 0) + rng.normal(0, 0.8), 18, 38
                )
                visibility = np.clip(
                    10 - (4 if rainy else 0) - (2 if weather.lower() == "foggy" else 0)
                    + rng.normal(0, 0.5), 0.5, 10
                )
                incidents = max(0, int(rng.poisson(max(0.02, values["Incident Reports"] * factor / 8))))
                congestion = np.clip(
                    100 * (1 - speed / 85) * 0.9
                    + (10 if roadwork.lower() == "yes" else 0)
                    + rng.normal(0, 3), 0, 100
                )
                capacity = np.clip(congestion + rng.normal(0, 8), 0, 100)
                travel_index = np.clip(1 + congestion / 100 * 0.8 + rng.normal(0, 0.025), 1, 1.8)
                environment = max(
                    0, values["Environmental Impact"] * (0.45 + 0.55 * factor)
                    + 0.12 * volume / 1000 + rng.normal(0, 2)
                )

                records.append({
                    "Timestamp": timestamp,
                    "Date": day.date().isoformat(),
                    "Area Name": area_by_road.get(road, "Unknown"),
                    "Road/Intersection Name": road,
                    "Traffic Volume": volume,
                    "Average Speed": round(float(speed), 3),
                    "Travel Time Index": round(float(travel_index), 4),
                    "Congestion Level": round(float(congestion), 3),
                    "Road Capacity Utilization": round(float(capacity), 3),
                    "Incident Reports": incidents,
                    "Environmental Impact": round(float(environment), 3),
                    "Public Transport Usage": round(float(np.clip(values["Public Transport Usage"] * factor, 0, 100)), 3),
                    "Traffic Signal Compliance": round(float(np.clip(values["Traffic Signal Compliance"] + rng.normal(0, 2), 0, 100)), 3),
                    "Parking Usage": round(float(np.clip(values["Parking Usage"] * factor, 0, 100)), 3),
                    "Pedestrian and Cyclist Count": max(0, int(values["Pedestrian and Cyclist Count"] * factor + rng.normal(0, 8))),
                    "Weather Conditions": weather,
                    "Roadwork and Construction Activity": roadwork,
                    "Hour": timestamp.hour,
                    "Minute": timestamp.minute,
                    "DayOfWeek": dow,
                    "IsWeekend": int(weekend),
                    "IsHoliday": int(holiday),
                    "Rainfall_mm": round(float(rainfall), 3),
                    "Temperature_C": round(float(temperature), 2),
                    "Visibility_km": round(float(visibility), 2),
                    "IncidentSeverity": min(3, incidents + int(incidents > 0 and rng.random() < 0.25)),
                    "DataType": "SYNTHETIC"
                })

    result = pd.DataFrame(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Source date range: {source['Date'].min().date()} to {end_date.date()}")
    print(f"Generated date range: {start_date.date()} to {end_date.date()}")
    print(f"Rows: {len(result):,}")
    print(f"Roads: {result['Road/Intersection Name'].nunique()}")
    print(f"Interval: {result.groupby('Road/Intersection Name')['Timestamp'].diff().dropna().value_counts().index[0]}")
    print(f"Output: {output_path.resolve()}")
    print("Reminder: generated intraday values are synthetic, not measured 5-minute observations.")

if __name__ == "__main__":
    main()
