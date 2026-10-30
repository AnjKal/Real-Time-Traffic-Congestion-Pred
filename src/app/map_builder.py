"""Base Folium map: 20 junction circles that the page's JavaScript recolors live."""
import folium
import pandas as pd

CENTER = (12.95, 77.65)
BAND_COLORS = {"free": "#2ecc71", "moderate": "#f1c40f", "heavy": "#e74c3c"}
UNKNOWN_COLOR = "#95a5a6"

LEGEND_HTML = """
<div style="position: fixed; bottom: 24px; right: 12px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #888; border-radius: 6px; font: 13px sans-serif;
            line-height: 1.6;">
  <b>Predicted speed vs free flow</b><br>
  <span style="color:#2ecc71">&#9679;</span> Free (&gt;75%)<br>
  <span style="color:#f1c40f">&#9679;</span> Moderate (40-75%)<br>
  <span style="color:#e74c3c">&#9679;</span> Heavy (&lt;40%)
</div>
"""


def node_color(band: str) -> str:
    return BAND_COLORS[band]


def build_base_map(nodes_df: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=CENTER, zoom_start=12, tiles="OpenStreetMap",
                   width="100%", height="100%")
    for row in nodes_df.itertuples():
        marker = folium.CircleMarker(
            location=(row.lat, row.lon), radius=10, color="#333333", weight=1,
            fill=True, fill_color=UNKNOWN_COLOR, fill_opacity=0.85,
            tooltip=f"{row.name} - waiting for forecast",
            class_name=f"node-{row.node_id}")             # DOM class: ".node-<id>"
        marker._node_id = int(row.node_id)
        marker.add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))
    return m


def node_var_names(m: folium.Map) -> dict[int, str]:
    """node_id -> the JavaScript variable Folium gave that circle (global in the page)."""
    return {child._node_id: child.get_name() for child in m._children.values()
            if hasattr(child, "_node_id")}
