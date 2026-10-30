"""Step 2.3: confirm each node snaps to a real road covered by TomTom.

Usage:
    python -m scripts.step2_verify_nodes          # all 20 nodes (20 calls)
    python -m scripts.step2_verify_nodes 10       # only node 10 (1 call), merged into the JSON
"""
import json
import sys
import time

import requests

from src.config import RAW_DIR, TOMTOM_API_KEY, TOMTOM_FLOW_URL
from src.nodes import NODES

OUT_PATH = RAW_DIR / "step2_tomtom_check.json"


def main() -> int:
    if not TOMTOM_API_KEY:
        print("TOMTOM_API_KEY not set in .env")
        return 1

    only = {int(a) for a in sys.argv[1:]}
    targets = [n for n in NODES if not only or n.node_id in only]

    results, failed, calls = [], [], 0
    print(f"{'id':>2} | {'name':<22} | {'road_class':<10} | {'free_flow':>9} | "
          f"{'confidence':>10} | status")
    for k, n in enumerate(targets):
        if k:
            time.sleep(1)
        params = {"point": f"{n.lat},{n.lon}", "unit": "KMPH", "openLr": "false",
                  "key": TOMTOM_API_KEY}
        reasons, seg = [], {}
        calls += 1
        try:
            r = requests.get(TOMTOM_FLOW_URL, params=params, timeout=15)
            if r.status_code != 200:
                reasons.append(f"HTTP {r.status_code}")
                payload = {"http_status": r.status_code, "body": r.text[:500]}
            else:
                payload = r.json()
                seg = payload.get("flowSegmentData", {})
        except requests.RequestException as e:
            reasons.append(f"request error: {e}")
            payload = {"error": str(e)}

        if not reasons:
            if seg.get("confidence", 0) < 0.5:
                reasons.append("confidence<0.5")
            if seg.get("freeFlowSpeed", 0) < 20:
                reasons.append("freeFlow<20")

        ok = not reasons
        if not ok:
            failed.append(n.node_id)
        results.append({
            "node_id": n.node_id, "name": n.name, "lat": n.lat, "lon": n.lon,
            "frc": seg.get("frc"), "currentSpeed": seg.get("currentSpeed"),
            "freeFlowSpeed": seg.get("freeFlowSpeed"), "confidence": seg.get("confidence"),
            "roadClosure": seg.get("roadClosure"),
            "road_name": seg.get("roadName") or seg.get("name"),
            "passed": ok, "fail_reasons": reasons, "payload": payload,
        })
        status = "PASS" if ok else "FAIL (" + ", ".join(reasons) + ")"
        print(f"{n.node_id:>2} | {n.name:<22} | {str(seg.get('frc', '-')):<10} | "
              f"{str(seg.get('freeFlowSpeed', '-')):>9} | "
              f"{str(seg.get('confidence', '-')):>10} | {status}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if only and OUT_PATH.exists():
        # partial run: replace just these nodes in the existing file
        merged = {r["node_id"]: r for r in json.loads(OUT_PATH.read_text())}
        merged.update({r["node_id"]: r for r in results})
        results = [merged[k] for k in sorted(merged)]
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nTomTom calls used: {calls}")
    print(f"Wrote {OUT_PATH}")
    if failed:
        print(f"FAILED node ids: {failed}")
        return 1
    print(f"All {len(targets)} checked node(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
