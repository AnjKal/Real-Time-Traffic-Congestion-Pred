"""Step 2.5: Folium map of the node graph."""
import csv
import pickle

import folium

from src.config import BBOX, GRAPH_DIR, PROJECT_ROOT

OUT = PROJECT_ROOT / "notebooks" / "step2_graph_viz.html"


def main() -> None:
    with open(GRAPH_DIR / "nodes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(GRAPH_DIR / "adj_mx.pkl", "rb") as f:
        _, _, W = pickle.load(f)

    m = folium.Map(location=[(BBOX["north"] + BBOX["south"]) / 2,
                             (BBOX["east"] + BBOX["west"]) / 2],
                   zoom_start=12, tiles="OpenStreetMap")
    folium.Rectangle([[BBOX["south"], BBOX["west"]], [BBOX["north"], BBOX["east"]]],
                     color="#3388ff", weight=1, fill=True, fill_opacity=0.05).add_to(m)

    for r in rows:
        folium.Marker(
            [float(r["lat"]), float(r["lon"])],
            popup=folium.Popup(f"{r['node_id']} | {r['name']} | {r['frc']} | "
                               f"{r['free_flow_speed_kmh']} km/h", max_width=300),
        ).add_to(m)

    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if W[i][j] > 0:
                folium.PolyLine(
                    [[float(rows[i]["lat"]), float(rows[i]["lon"])],
                     [float(rows[j]["lat"]), float(rows[j]["lon"])]],
                    weight=1 + 5 * W[i][j], opacity=0.6).add_to(m)

    OUT.parent.mkdir(exist_ok=True)
    m.save(str(OUT))
    print(f"Saved {OUT.resolve()}")


if __name__ == "__main__":
    main()
