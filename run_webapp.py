"""Start the web app (modern Python, NOT the TensorFlow 3.7 env).

    python run_webapp.py                 # http://127.0.0.1:8010/
    python run_webapp.py --port 8020
    set MODEL_SERVICE_URL=http://127.0.0.1:8001   # only if the model service runs elsewhere

Needs the model service (model_service.py) running, and the road network built once with
`python -m webapp.osm_graph`. Listens on 127.0.0.1 only.
"""
import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    print("Web app: http://127.0.0.1:%d/   (model service must be running on port 8001)" % args.port)
    uvicorn.run("webapp.server:app", host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
