"""Step 1 smoke test: environment, dependencies, directories, nodes, API key (no API call)."""
import importlib
import importlib.metadata as md
import sys

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

# (import name, distribution name)
PACKAGES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("sklearn", "scikit-learn"), ("torch", "torch"), ("osmnx", "osmnx"),
    ("networkx", "networkx"), ("shapely", "shapely"), ("geopy", "geopy"),
    ("requests", "requests"), ("pyarrow", "pyarrow"), ("dotenv", "python-dotenv"),
    ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("folium", "folium"),
    ("jinja2", "jinja2"), ("pytest", "pytest"), ("ruff", "ruff"),
    ("ipykernel", "ipykernel"),
]


def check_python() -> None:
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer required")


def check_imports() -> None:
    for mod, dist in PACKAGES:
        try:
            importlib.import_module(mod)
        except ImportError:
            # ruff is a CLI binary; the Python module may not be importable
            if mod != "ruff":
                raise RuntimeError(f"cannot import {mod} (pip install {dist})")
        print(f"  {dist}=={md.version(dist)}")


def check_dirs() -> None:
    from src import config

    for name in ("DATA_DIR", "RAW_DIR", "SYNTHETIC_DIR", "PROCESSED_DIR", "GRAPH_DIR",
                 "CHECKPOINT_DIR", "LOG_DIR"):
        path = getattr(config, name)
        if not path.is_dir():
            raise RuntimeError(f"missing directory {name}: {path}")


def check_nodes() -> None:
    from src.config import NUM_NODES
    from src.nodes import NODES

    if len(NODES) != NUM_NODES or len({n.node_id for n in NODES}) != NUM_NODES:
        raise RuntimeError(f"expected {NUM_NODES} nodes with unique ids")


def check_api_key() -> None:
    from src import config

    if not config.TOMTOM_API_KEY or config.TOMTOM_API_KEY == "your_key_here":
        raise RuntimeError("TOMTOM_API_KEY not set in .env")


def main() -> int:
    try:
        check_python()
        check_imports()
        check_dirs()
        check_nodes()
        check_api_key()
    except Exception as e:
        print(f"{RED}STEP 1 FAIL: {e}{RESET}")
        return 1
    print(f"{GREEN}STEP 1 OK{RESET}")
    from src import config

    if (config.GRAPH_DIR / "adj_mx.pkl").exists():
        print("Step 2 artifacts present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
