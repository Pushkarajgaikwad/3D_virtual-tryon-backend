"""
Entry point — run the Virtual Try-On server with plain Python.

Usage:
    python run.py              # default: port 8000
    python run.py --port 9000
    python run.py --reload     # auto-reload on code changes (dev mode)
"""

import argparse
import sys
import os

# Ensure project root is on the Python path regardless of where Python is invoked from
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="3D Virtual Try-On Server")
    parser.add_argument("--host",   default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true",  help="Auto-reload on code changes (dev)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed.")
        print("Fix:   pip install uvicorn[standard]")
        sys.exit(1)

    print(f"\n  3D Virtual Try-On Server")
    print(f"  ─────────────────────────────────────")
    print(f"  URL     : http://localhost:{args.port}")
    print(f"  API docs: http://localhost:{args.port}/docs")
    print(f"  Health  : http://localhost:{args.port}/api/health")
    print(f"  Reload  : {'yes (dev mode)' if args.reload else 'no'}")
    print(f"  ─────────────────────────────────────\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
