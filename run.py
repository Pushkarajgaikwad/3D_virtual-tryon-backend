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
import signal
import socket

# Ensure project root is on the Python path regardless of where Python is invoked from
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _kill_port(port: int) -> bool:
    """Kill any process currently listening on *port*. Returns True if something was killed."""
    import subprocess
    try:
        # lsof -t gives just the PIDs; works on macOS and Linux
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        if not pids:
            return False
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"  Killed existing process on port {port} (PID {pid})")
            except (ProcessLookupError, ValueError):
                pass
        return True
    except FileNotFoundError:
        # lsof not available — try fuser (Linux)
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            return True
        except FileNotFoundError:
            return False


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


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

    # Free the port if something is already bound to it
    if _port_in_use(args.host, args.port):
        print(f"  Port {args.port} is in use — attempting to free it ...")
        killed = _kill_port(args.port)
        if killed:
            import time
            time.sleep(1)          # give the OS a moment to release the port
        if _port_in_use(args.host, args.port):
            print(f"  ERROR: port {args.port} is still in use after kill attempt.")
            print(f"  Try: lsof -ti tcp:{args.port} | xargs kill -9")
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
