import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowRight, ArrowUpRight,
  Bell, CalendarDays, CarFront, Check, ChevronDown, CircleHelp,
  Clock3, CloudRain, Construction, Gauge, MapPin, Menu, Plus,
  RefreshCw, Search, Settings2, ShieldAlert, TrafficCone, Waves, X
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis
} from "recharts";
import {
  MapContainer, TileLayer, CircleMarker, Polyline, Popup
} from "react-leaflet";

const INTERVALS = [5, 10, 15, 20, 30, 60];
const ISSUE_TYPES = ["Construction", "Waterlogging", "Accident", "Road closure", "Pothole", "Signal failure"];
const API = "/api";

const demoRoads = [
  "100 Feet Road", "Anil Kumble Circle", "Ballari Road", "CMH Road",
  "Hebbal Flyover", "Hosur Road", "ITPL Main Road", "Jayanagar 4th Block",
  "Marathahalli Bridge", "Sarjapur Road", "Silk Board Junction",
  "Sony World Junction", "South End Circle", "Trinity Circle",
  "Tumkur Road", "Yeshwanthpur Circle"
];

const roadLocations = {
  "100 Feet Road": [12.9716, 77.6412],
  "Anil Kumble Circle": [12.9747, 77.6047],
  "Ballari Road": [13.0358, 77.5970],
  "CMH Road": [12.9784, 77.6408],
  "Hebbal Flyover": [13.0355, 77.5978],
  "Hosur Road": [12.9166, 77.6101],
  "ITPL Main Road": [12.9868, 77.7365],
  "Jayanagar 4th Block": [12.9250, 77.5838],
  "Marathahalli Bridge": [12.9591, 77.7010],
  "Sarjapur Road": [12.9104, 77.6830],
  "Silk Board Junction": [12.9177, 77.6233],
  "Sony World Junction": [12.9365, 77.6145],
  "South End Circle": [12.9368, 77.5802],
  "Trinity Circle": [12.9722, 77.6164],
  "Tumkur Road": [13.0285, 77.5402],
  "Yeshwanthpur Circle": [13.0238, 77.5511]
};

function classify(speed) {
  if (speed >= 40) return "free";
  if (speed >= 25) return "moderate";
  return "heavy";
}

function statusLabel(status) {
  return status === "free" ? "Free flow" : status === "moderate" ? "Moderate" : status === "heavy" ? "Heavy traffic" : "Unknown";
}

function fmtTime(value) {
  if (!value) return "—";
  return new Date(value).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
}

function App() {
  const [roads, setRoads] = useState([]);
  const [selectedRoad, setSelectedRoad] = useState("Silk Board Junction");
  const [interval, setInterval] = useState(5);
  const [dashboard, setDashboard] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [issues, setIssues] = useState([]);
  const [activeTab, setActiveTab] = useState("overview");
  const [query, setQuery] = useState("");
  const [showIssueForm, setShowIssueForm] = useState(false);
  const [issueForm, setIssueForm] = useState({ road: "Silk Board Junction", type: "Waterlogging", severity: "Moderate", description: "" });
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [apiError, setApiError] = useState("");
  const [notice, setNotice] = useState("");

  async function loadDashboard(quiet = false) {
    if (quiet) setRefreshing(true); else setLoading(true);
    try {
      const [dashRes, roadsRes, issuesRes] = await Promise.all([
        axios.get(`${API}/dashboard`),
        axios.get(`${API}/roads`),
        axios.get(`${API}/road-issues`)
      ]);
      setDashboard(dashRes.data);
      setRoads(roadsRes.data.roads || []);
      setIssues(issuesRes.data.issues || []);
      setApiError("");
    } catch (error) {
      setApiError(error.response?.data?.error || "Could not connect to the traffic API. Start the Flask backend and check that the model and dataset paths are correct.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  async function loadForecast(road) {
    try {
      const res = await axios.get(`${API}/predict/latest`, { params: { road } });
      setForecast(res.data);
    } catch (error) {
      setForecast(null);
      setApiError(error.response?.data?.error || "Unable to load forecast for this road.");
    }
  }

  useEffect(() => { loadDashboard(); }, []);
  useEffect(() => {
    if (selectedRoad) loadForecast(selectedRoad);
  }, [selectedRoad]);

  const roadRows = dashboard?.roads || [];
  const selected = roadRows.find(r => r.road === selectedRoad);
  const selectedIssues = issues.filter(i => i.road === selectedRoad && i.status !== "Resolved");
  const activeIssues = issues.filter(i => i.status !== "Resolved");
  const visibleRoads = roadRows.filter(r => r.road.toLowerCase().includes(query.toLowerCase()));
  const filteredForecast = useMemo(() => {
    const values = forecast?.predictions || [];
    return values.filter(p => p.horizon_minutes % interval === 0 || p.horizon_minutes === 60);
  }, [forecast, interval]);

  const summary = dashboard?.summary || {};
  const chartData = filteredForecast.map(p => ({
    time: `+${p.horizon_minutes}m`,
    speed: p.predicted_average_speed_kmh
  }));

  async function submitIssue(event) {
    event.preventDefault();
    try {
      await axios.post(`${API}/road-issues`, {
        ...issueForm,
        source: "Dashboard report"
      });
      setNotice("Road issue has been added to the incident board.");
      setShowIssueForm(false);
      setIssueForm({ road: selectedRoad, type: "Waterlogging", severity: "Moderate", description: "" });
      await loadDashboard(true);
      window.setTimeout(() => setNotice(""), 3500);
    } catch (error) {
      setNotice(error.response?.data?.error || "Could not save the road issue.");
      window.setTimeout(() => setNotice(""), 4000);
    }
  }

  function selectRoad(road) {
    setSelectedRoad(road);
    setActiveTab("overview");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={20} /></div>
          <div><div className="brand-name">CivicFlow</div><div className="brand-sub">BENGALURU TRAFFIC</div></div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list">
          <button className={`nav-item ${activeTab === "overview" ? "active" : ""}`} onClick={() => setActiveTab("overview")}><Gauge size={18} /> Overview</button>
          <button className={`nav-item ${activeTab === "roads" ? "active" : ""}`} onClick={() => setActiveTab("roads")}><MapPin size={18} /> Road network</button>
          <button className={`nav-item ${activeTab === "incidents" ? "active" : ""}`} onClick={() => setActiveTab("incidents")}><ShieldAlert size={18} /> Road issues <span className="nav-count">{activeIssues.length}</span></button>
        </nav>
        <div className="sidebar-bottom">
          <div className="system-card">
            <div className="system-top"><span className="live-dot" /> System status</div>
            <strong>{apiError ? "API disconnected" : "Prediction service"}</strong>
            <span className="system-caption">{apiError ? "Check backend connection" : "GRU model · Local API"}</span>
          </div>
          <div className="profile">
            <div className="avatar">BT</div>
            <div><strong>Traffic operations</strong><span>Local workspace</span></div>
            <Settings2 size={17} className="profile-settings" />
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>{activeTab === "overview" ? "Overview" : activeTab === "roads" ? "Road network" : "Road issues"}</strong></div>
          <div className="top-actions">
            <div className="location-chip"><MapPin size={14} /> Bengaluru, Karnataka</div>
            <button className="icon-button" aria-label="Notifications" onClick={() => setActiveTab("incidents")}><Bell size={18} />{activeIssues.length > 0 && <i />}</button>
            <div className="date-chip"><CalendarDays size={15} /> {new Date().toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" })}</div>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow">TRAFFIC OPERATIONS CENTRE <span className="eyebrow-line" /></div>
              <h1>{activeTab === "overview" ? "Bengaluru traffic overview" : activeTab === "roads" ? "Road network" : "Road issues & incidents"}</h1>
              <p className="subheading">Monitor road conditions, review short-term forecasts and track reported disruptions.</p>
            </div>
            <div className="heading-actions">
              <button className="button button-secondary" onClick={() => loadDashboard(true)} disabled={refreshing}><RefreshCw size={15} className={refreshing ? "spin" : ""} /> Refresh</button>
              <button className="button button-primary" onClick={() => { setIssueForm(f => ({ ...f, road: selectedRoad })); setShowIssueForm(true); }}><Plus size={16} /> Report road issue</button>
            </div>
          </section>

          {apiError && <div className="error-banner"><AlertTriangle size={17} /><div><strong>Backend connection issue</strong><span>{apiError}</span></div><button onClick={() => loadDashboard(true)}>Retry</button></div>}
          {notice && <div className="notice-banner"><Check size={16} /> {notice}</div>}

          {activeTab === "overview" && <>
            <section className="stats-grid">
              <StatCard label="Roads monitored" value={summary.total_roads ?? "—"} detail="Across the Bengaluru sample network" icon={<MapPin size={18} />} tone="blue" />
              <StatCard label="Free-flow roads" value={summary.free_roads ?? "—"} detail="Based on latest modelled speed" icon={<ArrowUpRight size={18} />} tone="green" />
              <StatCard label="Moderate traffic" value={summary.moderate_roads ?? "—"} detail="Slowdowns reported or estimated" icon={<CarFront size={18} />} tone="amber" />
              <StatCard label="Heavy traffic" value={summary.heavy_roads ?? "—"} detail="Requires closer attention" icon={<AlertTriangle size={18} />} tone="red" />
            </section>

            <div className="content-grid">
              <section className="panel network-panel">
                <div className="panel-header">
                  <div><h2>Bengaluru road map</h2><p>Monitored roads coloured by traffic status · Select a road for details</p></div>
                  <div className="map-label"><span className="map-dot" /> OpenStreetMap</div>
                </div>
                <TrafficMap roads={roadRows} selectedRoad={selectedRoad} onSelect={selectRoad} />
                <div className="map-legend">
                  <span><i className="legend-dot free" /> Free flow</span>
                  <span><i className="legend-dot moderate" /> Moderate</span>
                  <span><i className="legend-dot heavy" /> Heavy traffic</span>
                  <span><i className="legend-dot issue" /> Reported issue</span>
                </div>
              </section>

              <section className="panel road-panel">
                <div className="panel-header">
                  <div><h2>Road status</h2><p>Latest available observation</p></div>
                  <button className="text-button" onClick={() => setActiveTab("roads")}>View all <ArrowRight size={14} /></button>
                </div>
                <div className="road-search"><Search size={15} /><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search roads..." /></div>
                <div className="road-list">
                  {visibleRoads.slice(0, 7).map(r => <RoadRow key={r.road} road={r} selected={selectedRoad === r.road} onClick={() => selectRoad(r.road)} />)}
                  {!visibleRoads.length && <div className="empty-state">No roads match this search.</div>}
                </div>
              </section>
            </div>

            <div className="content-grid lower-grid">
              <section className="panel forecast-panel">
                <div className="panel-header forecast-header">
                  <div><h2>Short-term speed forecast</h2><p>{selectedRoad} · Predicted average speed</p></div>
                  <label className="interval-select"><Clock3 size={15} /><span>Interval</span><select value={interval} onChange={e => setInterval(Number(e.target.value))}>{INTERVALS.map(v => <option key={v} value={v}>{v} min</option>)}</select><ChevronDown size={14} /></label>
                </div>
                <div className="forecast-summary">
                  <div><span className="metric-label">Latest observed speed</span><strong>{selected?.latest_speed_kmh != null ? <>{selected.latest_speed_kmh.toFixed(1)} <span>km/h</span></> : "—"}</strong></div>
                  <div className="forecast-meta"><span>Forecast horizon</span><strong>Next 60 minutes</strong></div>
                  <div className="forecast-meta"><span>Last data point</span><strong>{fmtTime(selected?.last_observed_timestamp)}</strong></div>
                </div>
                {chartData.length ? <div className="chart-wrap"><ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={chartData} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
                    <defs><linearGradient id="speedFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#4b9b80" stopOpacity={0.22} /><stop offset="95%" stopColor="#4b9b80" stopOpacity={0.01} /></linearGradient></defs>
                    <CartesianGrid stroke="#edf0ef" vertical={false} />
                    <XAxis dataKey="time" tick={{ fill: "#87918e", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#87918e", fontSize: 11 }} axisLine={false} tickLine={false} unit=" km/h" />
                    <Tooltip contentStyle={{ border: "1px solid #e7ece9", borderRadius: 10, boxShadow: "0 8px 24px #233a2c12" }} formatter={v => [`${Number(v).toFixed(1)} km/h`, "Predicted speed"]} />
                    <Area type="monotone" dataKey="speed" stroke="#3d8b70" strokeWidth={2.5} fill="url(#speedFill)" activeDot={{ r: 5, strokeWidth: 2 }} />
                  </AreaChart>
                </ResponsiveContainer></div> : <div className="chart-empty"><Activity size={22} /><span>{loading ? "Loading forecast…" : "Select a road to view its forecast."}</span></div>}
                <div className="forecast-footnote">Forecasts are generated from the GRU model using the latest 12 five-minute observations. Display interval filters the forecast points.</div>
              </section>

              <section className="panel issues-panel">
                <div className="panel-header"><div><h2>Road issues</h2><p>Reported disruptions requiring attention</p></div><button className="text-button" onClick={() => setActiveTab("incidents")}>View all <ArrowRight size={14} /></button></div>
                <div className="issue-list">
                  {activeIssues.slice(0, 4).map(issue => <IssueRow key={issue.id} issue={issue} />)}
                  {!activeIssues.length && <div className="empty-issues"><div className="empty-icon"><Check size={19} /></div><strong>No active road issues</strong><span>Reports saved to PostgreSQL by operators will appear here.</span></div>}
                </div>
                <button className="report-link" onClick={() => setShowIssueForm(true)}><Plus size={15} /> Add a road issue</button>
              </section>
            </div>
          </>}

          {activeTab === "roads" && <section className="panel full-panel">
            <div className="panel-header"><div><h2>Monitored road network</h2><p>Choose a road to inspect its forecast and reported issues.</p></div><span className="small-pill">{roadRows.length} roads</span></div>
            <div className="table-wrap"><table><thead><tr><th>Road / junction</th><th>Traffic status</th><th>Avg. speed</th><th>Traffic volume</th><th>Active issues</th><th></th></tr></thead><tbody>
              {roadRows.map(r => <tr key={r.road} onClick={() => selectRoad(r.road)} className="click-row"><td><strong>{r.road}</strong><span className="cell-sub">{r.area || "Bengaluru"}</span></td><td><StatusPill status={r.status} /></td><td>{r.latest_speed_kmh?.toFixed(1) ?? "—"} km/h</td><td>{r.traffic_volume?.toLocaleString("en-IN") ?? "—"}</td><td>{r.active_issue_count || 0}</td><td><ArrowRight size={16} className="table-arrow" /></td></tr>)}
            </tbody></table></div>
          </section>}

          {activeTab === "incidents" && <section className="panel full-panel">
            <div className="panel-header"><div><h2>Reported road issues</h2><p>Operator-submitted reports. These are not automatically verified by a municipal authority.</p></div><button className="button button-primary" onClick={() => setShowIssueForm(true)}><Plus size={15} /> Report issue</button></div>
            {issues.length ? <div className="incident-grid">{issues.map(issue => <IssueCard key={issue.id} issue={issue} onResolve={async () => { await axios.patch(`${API}/road-issues/${issue.id}`, { status: "Resolved" }); loadDashboard(true); }} />)}</div> : <div className="empty-issues large"><div className="empty-icon"><Check size={20} /></div><strong>No issues have been reported</strong><span>Use “Report issue” to add construction, waterlogging, accidents or other road disruptions.</span></div>}
          </section>}

          <footer className="page-footer"><span>© 2026 CivicFlow · Traffic intelligence prototype</span><span><i className="footer-dot" /> {dashboard?.data_type === "SYNTHETIC" ? "Synthetic dataset" : "Model data"} · Not a live municipal feed</span></footer>
        </div>
      </main>

      {showIssueForm && <div className="modal-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) setShowIssueForm(false); }}>
        <form className="issue-modal" onSubmit={submitIssue}>
          <div className="modal-heading"><div><span className="eyebrow">FIELD REPORT</span><h2>Report a road issue</h2><p>Add a disruption that may affect traffic flow.</p></div><button type="button" className="icon-button close-button" onClick={() => setShowIssueForm(false)}><X size={18} /></button></div>
          <label>Road / junction<select value={issueForm.road} onChange={e => setIssueForm({ ...issueForm, road: e.target.value })}>{(roads.length ? roads : demoRoads).map(r => <option key={typeof r === "string" ? r : r} value={typeof r === "string" ? r : r}>{typeof r === "string" ? r : r}</option>)}</select></label>
          <div className="form-two"><label>Issue type<select value={issueForm.type} onChange={e => setIssueForm({ ...issueForm, type: e.target.value })}>{ISSUE_TYPES.map(t => <option key={t}>{t}</option>)}</select></label><label>Severity<select value={issueForm.severity} onChange={e => setIssueForm({ ...issueForm, severity: e.target.value })}><option>Low</option><option>Moderate</option><option>High</option></select></label></div>
          <label>Description<textarea rows="3" value={issueForm.description} onChange={e => setIssueForm({ ...issueForm, description: e.target.value })} placeholder="e.g. Water accumulation near the underpass after rainfall…" required /></label>
          <div className="modal-note"><CircleHelp size={15} /> Reports are stored in the project PostgreSQL database and should be verified before operational use.</div>
          <div className="modal-actions"><button type="button" className="button button-secondary" onClick={() => setShowIssueForm(false)}>Cancel</button><button className="button button-primary" type="submit"><Plus size={15} /> Submit report</button></div>
        </form>
      </div>}
    </div>
  );
}

function StatCard({ label, value, detail, icon, tone }) {
  return <div className="stat-card"><div className={`stat-icon ${tone}`}>{icon}</div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></div>;
}

function StatusPill({ status }) {
  return <span className={`status-pill ${status || "unknown"}`}><i />{statusLabel(status)}</span>;
}

function RoadRow({ road, selected, onClick }) {
  return <button className={`road-row ${selected ? "selected" : ""}`} onClick={onClick}>
    <span className={`road-indicator ${road.status || "unknown"}`} />
    <span className="road-row-main"><strong>{road.road}</strong><small>{road.latest_speed_kmh != null ? `${road.latest_speed_kmh.toFixed(1)} km/h` : "No speed data"}</small></span>
    <span className="road-row-end"><StatusPill status={road.status} /><ArrowRight size={14} /></span>
  </button>;
}

function IssueRow({ issue }) {
  const Icon = issue.type === "Waterlogging" ? Waves : issue.type === "Construction" ? Construction : issue.type === "Accident" ? AlertTriangle : TrafficCone;
  return <div className="issue-row"><div className={`issue-icon ${issue.severity?.toLowerCase()}`}><Icon size={17} /></div><div className="issue-main"><strong>{issue.type}</strong><span>{issue.road}</span></div><span className={`severity-tag ${issue.severity?.toLowerCase()}`}>{issue.severity}</span></div>;
}

function IssueCard({ issue, onResolve }) {
  const Icon = issue.type === "Waterlogging" ? Waves : issue.type === "Construction" ? Construction : issue.type === "Accident" ? AlertTriangle : TrafficCone;
  return <article className="incident-card"><div className="incident-card-top"><div className={`issue-icon ${issue.severity?.toLowerCase()}`}><Icon size={18} /></div><span className={`severity-tag ${issue.severity?.toLowerCase()}`}>{issue.severity}</span></div><h3>{issue.type}</h3><p className="incident-road"><MapPin size={13} /> {issue.road}</p><p className="incident-description">{issue.description}</p><div className="incident-card-bottom"><span>{issue.created_at ? new Date(issue.created_at).toLocaleString("en-IN") : "Recently reported"}</span>{issue.status !== "Resolved" ? <button onClick={onResolve}>Mark resolved <Check size={13} /></button> : <span className="resolved-label"><Check size={13} /> Resolved</span>}</div></article>;
}

function TrafficMap({ roads, selectedRoad, onSelect }) {
  const [roadGeometries, setRoadGeometries] = useState({});
  const [loadingGeometry, setLoadingGeometry] = useState(true);

  const roadByName = new Map(roads.map(road => [road.road, road]));

  useEffect(() => {
    let cancelled = false;

    const loadRoadGeometry = async () => {
      const roadNames = demoRoads
        .map(name => name.replace(/["\\]/g, ""))
        .join("|");

      const query = `
        [out:json][timeout:25];
        (
          way["name"~"${roadNames}",i](12.85,77.45,13.15,77.80);
        );
        out geom;
      `;

      try {
        const response = await fetch(
          "https://overpass.kumi.systems/api/interpreter",
          {
            method: "POST",
            headers: {
              "Content-Type": "text/plain"
            },
            body: query
          }
        );

        if (!response.ok) {
          throw new Error("Unable to retrieve OpenStreetMap geometry");
        }

        const result = await response.json();

        console.log("Overpass response:", result);
        console.log("Number of road geometries:", result.elements?.length);
        console.log(
          "Road names found:",
          [...new Set(result.elements.map(e => e.tags?.name).filter(Boolean))]
        );

        const geometries = {};

        result.elements.forEach(element => {
          if (!element.tags?.name || !element.geometry) return;

          const roadNameAliases = {
            "CMH Road": ["CMH Road", "CMH"],
            "Jayanagar 4th Block": ["Jayanagar 4th Block", "Jayanagar 4th"],
            "Silk Board Junction": ["Silk Board Junction", "Silk Board"],
            "Sony World Junction": ["Sony World Junction", "Sony World"],
            "Yeshwanthpur Circle": ["Yeshwanthpur Circle", "Yeshwanthpur"]
          };

          const matchedRoad = demoRoads.find(name =>
            (roadNameAliases[name] || [name]).some(alias =>
              element.tags.name.toLowerCase().includes(alias.toLowerCase())
            )
          );

          if (!matchedRoad) return;

          if (!geometries[matchedRoad]) {
            geometries[matchedRoad] = [];
          }

          geometries[matchedRoad].push(
            element.geometry.map(point => [point.lat, point.lon])
          );
        });

        if (!cancelled) {
          setRoadGeometries(geometries);

          const matched = Object.keys(geometries);
          const unmatched = demoRoads.filter(name => !geometries[name]);

          console.log("Matched roads:", matched);
          console.log("Unmatched roads:", unmatched);
          console.log("Total matched:", matched.length);
        }
      } catch (error) {
        console.error("Road geometry loading failed:", error);
      } finally {
        if (!cancelled) {
          setLoadingGeometry(false);
        }
      }
    };

    loadRoadGeometry();

    return () => {
      cancelled = true;
    };
  }, []);

  const markerColor = road => {
    if (!road) return "#98a39d";
    if (road.status === "free") return "#39956f";
    if (road.status === "moderate") return "#d99a32";
    if (road.status === "heavy") return "#d75b55";
    return "#98a39d";
  };

  const markers = demoRoads.map(name => ({
    name,
    position: roadLocations[name],
    data: roadByName.get(name)
  }));

  return (
    <div className="real-map-wrap">
      <MapContainer
        center={[12.9716, 77.5946]}
        zoom={11}
        scrollWheelZoom={true}
        className="real-traffic-map"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {markers.map(marker => {
          const road = marker.data;
          const color = markerColor(road);
          const segments = roadGeometries[marker.name] || [];

          return (
            <React.Fragment key={marker.name}>
              {segments.map((positions, index) => (
                <Polyline
                  key={`${marker.name}-${index}`}
                  positions={positions}
                  pathOptions={{
                    color,
                    weight: selectedRoad === marker.name ? 8 : 6,
                    opacity: 0.9
                  }}
                  eventHandlers={{
                    click: () => onSelect(marker.name)
                  }}
                >
                  <Popup>
                    <div className="map-popup">
                      <strong>{marker.name}</strong>
                      <span>Status: {statusLabel(road?.status)}</span>
                      <span>
                        Average speed:{" "}
                        {road?.latest_speed_kmh != null
                          ? `${Number(road.latest_speed_kmh).toFixed(1)} km/h`
                          : "No reading available"}
                      </span>
                      <span>
                        Traffic volume:{" "}
                        {road?.traffic_volume != null
                          ? Number(road.traffic_volume).toLocaleString("en-IN")
                          : "No reading available"}
                      </span>
                      <button
                        type="button"
                        onClick={() => onSelect(marker.name)}
                      >
                        View road forecast
                      </button>
                    </div>
                  </Popup>
                </Polyline>
              ))}

              {segments.length === 0 && (
                <CircleMarker
                  center={marker.position}
                  radius={selectedRoad === marker.name ? 10 : 7}
                  pathOptions={{
                    color: "#ffffff",
                    weight: 2,
                    fillColor: color,
                    fillOpacity: 0.95
                  }}
                  eventHandlers={{
                    click: () => onSelect(marker.name)
                  }}
                >
                  <Popup>
                    <div className="map-popup">
                      <strong>{marker.name}</strong>
                      <span>Status: {statusLabel(road?.status)}</span>
                      <span>
                        Average speed:{" "}
                        {road?.latest_speed_kmh != null
                          ? `${Number(road.latest_speed_kmh).toFixed(1)} km/h`
                          : "No reading available"}
                      </span>
                      <button
                        type="button"
                        onClick={() => onSelect(marker.name)}
                      >
                        View road forecast
                      </button>
                    </div>
                  </Popup>
                </CircleMarker>
              )}
            </React.Fragment>
          );
        })}
      </MapContainer>

      <div className="map-data-note">
        {loadingGeometry
          ? "Loading road geometries from OpenStreetMap..."
          : "Road geometries sourced from OpenStreetMap. Unmatched roads are shown as markers."}
      </div>
    </div>
  );
}

export default App;