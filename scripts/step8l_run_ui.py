"""Step 8-lite: start the local map UI at http://127.0.0.1:<port>/ (localhost only)."""
import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    print(f"UI live at http://127.0.0.1:{args.port}/")
    print("Make sure the poller is running in another terminal so predictions stay fresh.")
    uvicorn.run("src.app.server:app", host="127.0.0.1", port=args.port, reload=False,
                log_level="info")


if __name__ == "__main__":
    main()
