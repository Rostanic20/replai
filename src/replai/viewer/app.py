from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..context import get_store

STATIC = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="replai", docs_url=None, redoc_url=None)

    @app.get("/api/runs")
    def list_runs():
        return get_store().runs()

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        store = get_store()
        return {"run": store.run(run_id), "spans": store.spans(run_id)}

    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
    return app


app = create_app()
