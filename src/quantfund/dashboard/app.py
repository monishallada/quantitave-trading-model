"""Dashboard API + static frontend. No auth: this is a localhost research tool
bound to 127.0.0.1 — do not expose it to the internet as-is."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from quantfund.core.state import PlatformState

_STATIC = Path(__file__).parent / "static"


class KillRequest(BaseModel):
    reason: str = "dashboard"


def create_app(state: PlatformState,
               kill: Optional[Callable[[str], Any]] = None,
               release: Optional[Callable[[], Any]] = None) -> FastAPI:
    app = FastAPI(title="quantfund dashboard", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": state.mode}

    @app.get("/api/summary")
    def summary():
        return state.snapshot_view()

    @app.get("/api/trades")
    def trades(limit: int = 200):
        return state.get_trades(limit=min(limit, 1000))

    @app.get("/api/rationale/{rationale_id}")
    def rationale(rationale_id: str):
        r = state.get_rationale(rationale_id)
        if r is None:
            raise HTTPException(status_code=404, detail="rationale not found")
        return r

    @app.get("/api/equity")
    def equity():
        return state.get_equity_curve()

    @app.get("/api/events")
    def events(limit: int = 100):
        return state.get_events(limit=min(limit, 500))

    @app.post("/api/kill")
    def kill_endpoint(body: KillRequest):
        if kill is not None:
            kill(body.reason)
        else:
            state.engage_kill_switch(body.reason)
        return {"ok": True}

    @app.post("/api/release")
    def release_endpoint():
        if release is not None:
            release()
        else:
            state.release_kill_switch()
        return {"ok": True}

    return app
