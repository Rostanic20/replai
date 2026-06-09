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

    diff = sub.add_parser("diff", help="Compare two runs step by step")
    diff.add_argument("run_a")
    diff.add_argument("run_b")
    diff.add_argument("--db", default=None)

    sub.add_parser("version", help="Show the installed version")

    args = parser.parse_args(argv)

    if args.command == "ui":
        _launch_ui(args.host, args.port, args.db)
    elif args.command == "runs":
        _list_runs()
    elif args.command == "diff":
        _diff(args.run_a, args.run_b, args.db)
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


def _diff(run_a: str, run_b: str, db) -> None:
    from .diff import diff_runs, summarize
    from .store import Store
    store = Store(db)
    spans_a, spans_b = store.spans(run_a), store.spans(run_b)
    if not spans_a and not store.run(run_a):
        raise SystemExit(f"No run found: {run_a}")
    if not spans_b and not store.run(run_b):
        raise SystemExit(f"No run found: {run_b}")

    sym = {"same": "  ", "changed": "~ ", "added": "+ ", "removed": "- "}
    print(f"A {run_a}   B {run_b}\n")
    for row in diff_runs(spans_a, spans_b):
        span = row["a"] or row["b"]
        print(f"{sym[row['status']]}{span['type']:<10} {span['name']}")
        for field, (old, new) in row["changes"].items():
            print(f"      {field}: {_short(old)} -> {_short(new)}")

    sa, sb = summarize(spans_a), summarize(spans_b)
    print(f"\n  steps {sa['steps']}->{sb['steps']}   "
          f"tokens_in {sa['tokens_in']}->{sb['tokens_in']}   "
          f"tokens_out {sa['tokens_out']}->{sb['tokens_out']}   "
          f"errors {sa['errors']}->{sb['errors']}")


def _short(value, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _ago(ts) -> str:
    if not ts:
        return ""
    secs = int(time.time() - ts)
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= n:
            return f"{secs // n}{unit} ago"
    return f"{secs}s ago"
