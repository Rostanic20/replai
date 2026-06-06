from __future__ import annotations

import argparse
import time


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="replai", description="Local-first debugger for LLM agents."
    )
    sub = parser.add_subparsers(dest="command")

    ui = sub.add_parser("ui", help="Launch the local viewer")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8473)
    ui.add_argument("--db", default=None)

    sub.add_parser("runs", help="List recorded runs")
    sub.add_parser("version", help="Show the installed version")

    args = parser.parse_args(argv)

    if args.command == "ui":
        _launch_ui(args.host, args.port, args.db)
    elif args.command == "runs":
        _list_runs()
    elif args.command == "version":
        from . import __version__
        print(__version__)
    else:
        parser.print_help()


def _launch_ui(host: str, port: int, db) -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("The viewer needs extra deps. Install with:  pip install 'replai[viewer]'")
    from . import context as _ctx
    from .store import Store
    _ctx.set_store(Store(db))
    from .viewer.app import app
    print(f"replai viewer → http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _list_runs() -> None:
    from .store import Store
    runs = Store().runs()
    if not runs:
        print("No runs recorded yet. Instrument your agent, run it, then try again.")
        return
    for r in runs:
        print(f"{r['id']}  {r['name']:<24}  {_ago(r['start'])}")


def _ago(ts) -> str:
    if not ts:
        return ""
    secs = int(time.time() - ts)
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= n:
            return f"{secs // n}{unit} ago"
    return f"{secs}s ago"
