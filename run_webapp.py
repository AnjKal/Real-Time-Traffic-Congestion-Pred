"""Start the web app (modern Python, NOT the TensorFlow 3.7 env).

    python run_webapp.py                 # http://127.0.0.1:8010/
    python run_webapp.py --port 8020
    set MODEL_SERVICE_URL=http://127.0.0.1:8001   # only if the model service runs elsewhere

The normal live setup is Docker Compose (see README.md). The road graph must be built once with
`python -m webapp.osm_graph`. The default host is loopback; Docker passes `--host 0.0.0.0`.
"""
import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP bind address; use 0.0.0.0 inside Docker")
    args = parser.parse_args()
    print("Web app: http://%s:%d/   (model service must be running on port 8001)" % (
        args.host, args.port))
    uvicorn.run("webapp.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
