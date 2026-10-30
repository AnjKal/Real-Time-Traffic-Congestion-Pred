from datetime import date, datetime, timedelta

import numpy as np
import pytest

from src.nodes import NODES
from src.tomtom import quota, storage
from src.tomtom.buffer import RollingBuffer
from src.tomtom.client import ClientError, RateLimitError, TomTomClient, TransientError
from src.tomtom.poller import Poller


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _payload(speed=30.0):
    return {"frc": "FRC2", "currentSpeed": speed, "freeFlowSpeed": 40.0, "confidence": 0.9,
            "roadClosure": False}


# ------------------------------------------------------------------ client
def test_client_parses_200(monkeypatch):
    c = TomTomClient("k")
    monkeypatch.setattr(c._session, "get",
                        lambda *a, **k: FakeResponse(200, {"flowSegmentData": _payload(33)}))
    assert c.fetch_point(12.9, 77.6)["currentSpeed"] == 33


@pytest.mark.parametrize("status,exc", [(429, RateLimitError), (500, TransientError),
                                        (503, TransientError), (403, ClientError),
                                        (400, ClientError)])
def test_client_error_mapping(monkeypatch, status, exc):
    c = TomTomClient("k")
    monkeypatch.setattr(c._session, "get", lambda *a, **k: FakeResponse(status))
    with pytest.raises(exc):
        c.fetch_point(12.9, 77.6)


def test_client_timeout_is_transient_and_hides_key(monkeypatch):
    import requests

    c = TomTomClient("SECRETKEY")

    def boom(*a, **k):
        raise requests.Timeout("https://x?key=SECRETKEY timed out")

    monkeypatch.setattr(c._session, "get", boom)
    with pytest.raises(TransientError) as e:
        c.fetch_point(12.9, 77.6)
    assert "SECRETKEY" not in str(e.value)


# ------------------------------------------------------------------ buffer
def test_buffer_push_shifts_and_normalizes():
    b = RollingBuffer(num_nodes=2, seq_len=3)
    ff = np.array([40.0, 20.0])
    b.push(datetime(2026, 10, 30, 6, 30), np.array([20.0, 40.0]), ff)
    b.push(datetime(2026, 10, 30, 6, 33), np.array([40.0, 10.0]), ff)
    data, ts = b.snapshot()
    assert data.shape == (3, 2, 2)
    assert np.isnan(data[0]).all()                       # still warming up
    np.testing.assert_allclose(data[1, :, 0], [0.5, 1.5])  # 40/20=2.0 clamped to 1.5
    np.testing.assert_allclose(data[2, :, 0], [1.0, 0.5])
    np.testing.assert_allclose(data[2, :, 1], (6 + 33 / 60) / 24, rtol=1e-6)
    assert ts[0] is None and ts[2] == datetime(2026, 10, 30, 6, 33)


def test_buffer_ready_after_seq_len_pushes():
    b = RollingBuffer(num_nodes=2, seq_len=12)
    for i in range(12):
        assert not b.is_ready()
        b.push(datetime(2026, 10, 30, 8, i), np.array([30.0, 30.0]), np.array([40.0, 40.0]))
    assert b.is_ready()


def test_buffer_nan_speed_keeps_it_not_ready():
    b = RollingBuffer(num_nodes=2, seq_len=1)
    b.push(datetime(2026, 10, 30, 8, 0), np.array([np.nan, 30.0]), np.array([40.0, 40.0]))
    assert not b.is_ready()


def test_buffer_save_load_roundtrip(tmp_path):
    b = RollingBuffer(num_nodes=3, seq_len=4)
    for i in range(3):
        b.push(datetime(2026, 10, 30, 9, i), np.array([10.0, 20.0, 30.0]), np.full(3, 40.0))
    b.save(tmp_path / "buf.pkl")
    r = RollingBuffer.load(tmp_path / "buf.pkl")
    (d1, t1), (d2, t2) = b.snapshot(), r.snapshot()
    np.testing.assert_array_equal(d1, d2)
    assert t1 == t2 and r.is_ready() == b.is_ready()


# ------------------------------------------------------------------ storage / quota
def test_storage_partition_roundtrip(tmp_path):
    now = datetime(2026, 10, 30, 12, 0)
    storage.append_observation(now, 3, 12.9, 77.6, 25.0, 40.0, 0.9, False, "FRC2",
                               "x" * 5000, store_dir=tmp_path)
    storage.append_observation(now, 4, 12.9, 77.6, 26.0, 40.0, 0.8, False, "FRC2", "{}",
                               store_dir=tmp_path)
    assert (tmp_path / "date=2026-10-30" / "observations.parquet").exists()
    df = storage.read_last_hours(1, store_dir=tmp_path, now=datetime(2026, 10, 30, 12, 30))
    assert list(df.columns) == storage.COLUMNS and len(df) == 2
    assert df["raw_payload"].str.len().max() == 2048
    assert storage.read_last_hours(1, store_dir=tmp_path, now=datetime(2026, 10, 30, 14, 0)).empty


def test_quota_resets_on_new_day(tmp_path):
    p = tmp_path / "q.json"
    quota.increment(500, p, today=date(2026, 10, 29))
    assert quota.get_calls_used_today(p, today=date(2026, 10, 29)) == 500
    assert quota.get_calls_used_today(p, today=date(2026, 10, 30)) == 0
    quota.increment(20, p, today=date(2026, 10, 30))
    assert quota.get_calls_used_today(p, today=date(2026, 10, 30)) == 20


# ------------------------------------------------------------------ poller
class FakeClient:
    """behaviours: node lat -> list of outcomes consumed per call (exception or payload)."""

    def __init__(self, behaviours=None):
        self.behaviours = {k: list(v) for k, v in (behaviours or {}).items()}
        self.calls = 0

    def fetch_point(self, lat, lon):
        self.calls += 1
        queue = self.behaviours.get(lat)
        outcome = queue.pop(0) if queue else _payload()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _poller(client, tmp_path, **kw):
    return Poller(client, RollingBuffer(), NODES, 3, None, None, backoff_s=0.0,
                  free_flow_ref=np.full(20, 40.0), buffer_path=None, store_dir=tmp_path / "s",
                  quota_path=tmp_path / "q.json", sleep_fn=lambda s: None,
                  now_fn=lambda: datetime(2026, 10, 30, 9, 0, 0), **kw)


def test_poll_once_all_succeed(tmp_path):
    p = _poller(FakeClient(), tmp_path)
    s = p.poll_once()
    assert (s["ok"], s["failed"], s["calls_used"]) == (20, 0, 20)
    assert quota.get_calls_used_today(tmp_path / "q.json") == 20
    data, _ = p.buffer.snapshot()
    np.testing.assert_allclose(data[-1, :, 0], 30.0 / 40.0)


def test_poll_once_retries_500_then_succeeds(tmp_path):
    flaky = {NODES[3].lat: [TransientError("HTTP 500")], NODES[7].lat: [TransientError("HTTP 500")]}
    client = FakeClient(flaky)
    s = _poller(client, tmp_path).poll_once()
    assert (s["ok"], s["failed"], s["calls_used"]) == (20, 0, 22)
    assert client.calls == 22


def test_poll_once_permanent_failure_records_nan_and_continues(tmp_path):
    dead = {NODES[5].lat: [TransientError("HTTP 500")] * 10}
    p = _poller(FakeClient(dead), tmp_path, max_retries=2)
    s = p.poll_once()
    assert (s["ok"], s["failed"], s["calls_used"]) == (19, 1, 22)       # 3 attempts on node 5
    df = storage.read_last_hours(1, store_dir=tmp_path / "s", now=datetime(2026, 10, 30, 9, 30))
    assert len(df) == 20 and df.loc[df.node_id == 5, "speed_kmh"].isna().all()
    assert not p.buffer.is_ready()


def test_poll_once_respects_quota(tmp_path):
    quota.increment(2395, tmp_path / "q.json")
    s = _poller(FakeClient(), tmp_path).poll_once()
    assert s["calls_used"] == 5 and s["ok"] == 5


def test_scheduling_helpers(tmp_path):
    p = Poller(FakeClient(), RollingBuffer(), NODES, 7, "08:00", "22:00",
               free_flow_ref=np.full(20, 40.0), buffer_path=None)
    assert p._in_window(datetime(2026, 10, 30, 8, 0)) and not p._in_window(datetime(2026, 10, 30, 7, 59))
    assert not p._in_window(datetime(2026, 10, 30, 22, 1))
    assert p._seconds_until_window_start(datetime(2026, 10, 30, 7, 0)) == 3600
    assert p._seconds_until_window_start(datetime(2026, 10, 30, 23, 0)) == 9 * 3600
    # boundaries are multiples of 7 min from 08:00: 08:00, 08:07, 08:14 ...
    assert p._seconds_until_next_boundary(datetime(2026, 10, 30, 8, 0, 5)) == 7 * 60 - 5
    assert p._seconds_until_next_boundary(datetime(2026, 10, 30, 8, 7, 0)) == 7 * 60


def test_run_forever_saves_buffer_on_ctrl_c(tmp_path):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    p = Poller(FakeClient(), RollingBuffer(), NODES, 3, None, None, free_flow_ref=np.full(20, 40.0),
               buffer_path=tmp_path / "buf.pkl", store_dir=tmp_path / "s",
               quota_path=tmp_path / "q.json", sleep_fn=interrupt,
               now_fn=lambda: datetime(2026, 10, 30, 9, 0, 0))
    p.run_forever()                                      # polls once, then "Ctrl-C" during sleep
    restored = RollingBuffer.load(tmp_path / "buf.pkl")
    assert not np.isnan(restored.snapshot()[0][-1]).any()


def test_run_forever_sleeps_to_midnight_when_quota_hit(tmp_path):
    quota.increment(2395, tmp_path / "q.json")
    slept = []

    def record_then_stop(seconds):
        slept.append(seconds)
        raise KeyboardInterrupt

    client = FakeClient()
    p = Poller(client, RollingBuffer(), NODES, 3, None, None, free_flow_ref=np.full(20, 40.0),
               buffer_path=None, store_dir=tmp_path / "s", quota_path=tmp_path / "q.json",
               sleep_fn=record_then_stop, now_fn=lambda: datetime(2026, 10, 30, 23, 0, 0))
    p.run_forever()
    assert client.calls == 0 and slept == [3601]


def _warmup_poller(client, tmp_path):
    clock = {"t": datetime(2026, 10, 30, 11, 0, 0)}

    def sleep(seconds):                      # advance the fake clock instead of waiting
        clock["t"] += timedelta(seconds=seconds)

    p = Poller(client, RollingBuffer(), NODES, 3, None, None, max_retries=0, backoff_s=0.0,
               free_flow_ref=np.full(20, 40.0), buffer_path=tmp_path / "buf.pkl",
               store_dir=tmp_path / "s", quota_path=tmp_path / "q.json",
               sleep_fn=sleep, now_fn=lambda: clock["t"])
    return p


def test_warmup_stops_after_exactly_seq_len_successful_polls(tmp_path):
    client = FakeClient()
    p = _warmup_poller(client, tmp_path)
    lines = []
    assert p.run_warmup(status_fn=lines.append) is True
    assert client.calls == 12 * 20                       # no polls beyond the 12th
    assert p.buffer.is_ready()
    assert RollingBuffer.load(tmp_path / "buf.pkl").is_ready()
    assert lines[0].startswith("[warmup] poll 1/12 at 11:00:00  ok=20 failed=0  calls_used=20")
    assert lines[-1].startswith("WARMUP COMPLETE: 12 timesteps buffered")


def test_warmup_does_not_count_failed_polls(tmp_path):
    # first poll: every node fails; second poll: one node fails; then all good
    behaviours = {n.lat: [TransientError("HTTP 500")] for n in NODES}
    behaviours[NODES[2].lat] = [TransientError("x"), TransientError("x")]
    client = FakeClient(behaviours)
    p = _warmup_poller(client, tmp_path)
    lines = []
    assert p.run_warmup(status_fn=lines.append) is True
    assert sum("(not counted)" in l for l in lines) == 2
    assert client.calls == 14 * 20                       # 2 bad polls + 12 valid ones
    _, stamps = p.buffer.snapshot()
    assert stamps[-1] - stamps[0] == timedelta(minutes=33)   # 12 consecutive valid polls


def test_warmup_gives_up_after_repeated_total_failures(tmp_path):
    client = FakeClient({n.lat: [TransientError("x")] * 50 for n in NODES})
    p = _warmup_poller(client, tmp_path)
    assert p.run_warmup(status_fn=lambda l: None, max_consecutive_bad=3) is False
    assert client.calls == 3 * 20 and not (tmp_path / "buf.pkl").exists()
