"""Thin TomTom Flow Segment client. Error text never includes the URL (it carries the API key)."""
import requests

from src.config import TOMTOM_FLOW_URL


class RateLimitError(Exception):
    """HTTP 429."""


class TransientError(Exception):
    """HTTP 5xx, timeout or connection problem - worth retrying."""


class ClientError(Exception):
    """Anything else (bad key, bad request, malformed body) - retrying will not help."""


class TomTomClient:
    def __init__(self, api_key: str, timeout_s: float = 10.0):
        if not api_key:
            raise ValueError("TomTom API key is required")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._session = requests.Session()

    def fetch_point(self, lat: float, lon: float) -> dict | None:
        params = {"point": f"{lat},{lon}", "unit": "KMPH", "openLr": "false",
                  "key": self._api_key}
        try:
            r = self._session.get(TOMTOM_FLOW_URL, params=params, timeout=self._timeout_s)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise TransientError(f"network error: {type(e).__name__}") from None
        except requests.RequestException as e:
            raise ClientError(f"request failed: {type(e).__name__}") from None

        if r.status_code == 200:
            try:
                return r.json()["flowSegmentData"]
            except (ValueError, KeyError):
                raise ClientError("HTTP 200 but no flowSegmentData in body") from None
        if r.status_code == 429:
            raise RateLimitError("HTTP 429")
        if 500 <= r.status_code < 600:
            raise TransientError(f"HTTP {r.status_code}")
        raise ClientError(f"HTTP {r.status_code}")
