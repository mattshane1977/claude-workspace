"""
FastAPI web server — dashboard, REST API, WebSocket log streaming,
settings management, and trader subprocess control.

Run standalone:  uvicorn web.app:app --host 0.0.0.0 --port 8080 --reload
Or via start.sh which launches everything.
"""
import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

# Ensure project root is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import db  # noqa: E402  (after sys.path fix)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Trader", docs_url="/api/docs")

STATIC_DIR = Path(__file__).parent / "static"
ENV_FILE = ROOT / ".env"
LOGS_DIR = ROOT / "logs"
TRADER_LOG = LOGS_DIR / "trader.log"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Trader process management ─────────────────────────────────────────────────
_trader_proc: Optional[subprocess.Popen] = None
_log_subscribers: list[asyncio.Queue] = []


def _broadcast_log(line: str):
    for q in _log_subscribers:
        try:
            q.put_nowait(line)
        except asyncio.QueueFull:
            pass


def trader_status() -> dict:
    global _trader_proc
    if _trader_proc is None:
        return {"running": False, "pid": None}
    if _trader_proc.poll() is None:
        return {"running": True, "pid": _trader_proc.pid}
    _trader_proc = None
    return {"running": False, "pid": None}


def start_trader():
    global _trader_proc
    if trader_status()["running"]:
        return
    LOGS_DIR.mkdir(exist_ok=True)
    _trader_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py")],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    # Background thread to read process output and broadcast to WS clients
    import threading

    def _read_output():
        for line in iter(_trader_proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                _broadcast_log(line)
        _trader_proc.stdout.close()

    t = threading.Thread(target=_read_output, daemon=True)
    t.start()
    logger.info(f"Trader started (pid={_trader_proc.pid})")


def stop_trader():
    global _trader_proc
    if _trader_proc and _trader_proc.poll() is None:
        _trader_proc.terminate()
        try:
            _trader_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _trader_proc.kill()
        _trader_proc = None
        logger.info("Trader stopped")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    db.init()
    logger.info("Web server started. Trader not auto-started — use the UI.")


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── API: status & control ─────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    status = trader_status()
    cycles = db.get_cycles(limit=1)
    last_cycle = cycles[0] if cycles else None
    return {**status, "last_cycle": last_cycle}


@app.post("/api/trader/start")
async def api_start():
    start_trader()
    return trader_status()


@app.post("/api/trader/stop")
async def api_stop():
    stop_trader()
    return trader_status()


@app.post("/api/trader/cycle")
async def api_run_cycle():
    """Manually trigger one decision cycle (runs in background thread)."""
    import threading

    def _run():
        import importlib
        # Re-import to get fresh config
        import config as cfg
        importlib.reload(cfg)
        from brain import context_builder, llm_agent, decision_parser
        from execution import alpaca_client, order_manager

        try:
            if not alpaca_client.is_market_open():
                _broadcast_log("Market is closed — manual cycle skipped.")
                return
            context = context_builder.build(cfg.WATCHLIST)
            portfolio = context.get("portfolio", {})
            raw = llm_agent.decide(context_builder.to_prompt_text(context))
            if not raw:
                return
            actions = decision_parser.parse(raw)
            if not actions:
                return
            cycle_id = db.insert_cycle(portfolio, raw.get("summary", ""))
            db.insert_decisions(cycle_id, actions)
            results = order_manager.execute(actions)
            db.insert_trades(cycle_id, results)
            _broadcast_log("Manual cycle complete.")
        except Exception as e:
            _broadcast_log(f"Manual cycle error: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"queued": True}


# ── API: data ─────────────────────────────────────────────────────────────────
@app.get("/api/portfolio")
async def api_portfolio():
    try:
        import config as cfg
        from risk.portfolio_state import get_state
        from execution.alpaca_client import is_market_open, get_open_orders
        state = get_state()
        state["market_open"] = is_market_open()
        state["open_orders"] = get_open_orders()
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cycles")
async def api_cycles(limit: int = 50):
    return db.get_cycles(limit)


@app.get("/api/decisions")
async def api_decisions(limit: int = 100, symbol: str = None):
    return db.get_decisions(limit, symbol)


@app.get("/api/trades")
async def api_trades(limit: int = 100, symbol: str = None):
    return db.get_trades(limit, symbol)


@app.get("/api/equity-history")
async def api_equity_history(limit: int = 200):
    return db.get_equity_history(limit)


# ── API: settings ─────────────────────────────────────────────────────────────
def _read_env() -> dict:
    """Parse .env file into a dict."""
    if not ENV_FILE.exists():
        return {}
    result = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def _write_env(data: dict):
    """Write dict back to .env (preserves comments, updates existing keys)."""
    existing_lines = []
    if ENV_FILE.exists():
        existing_lines = ENV_FILE.read_text().splitlines()

    written_keys = set()
    new_lines = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in data:
                new_lines.append(f"{k}={data[k]}")
                written_keys.add(k)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append any new keys not already in file
    for k, v in data.items():
        if k not in written_keys:
            new_lines.append(f"{k}={v}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n")


MASKED_KEYS = {"ALPACA_API_KEY", "ALPACA_SECRET_KEY"}


@app.get("/api/settings")
async def api_get_settings():
    env = _read_env()
    result = {}
    for k, v in env.items():
        result[k] = ("*" * 8 + v[-4:]) if k in MASKED_KEYS and len(v) > 4 else v
    return result


class SettingsUpdate(BaseModel):
    ALPACA_API_KEY: Optional[str] = None
    ALPACA_SECRET_KEY: Optional[str] = None
    ALPACA_BASE_URL: Optional[str] = None
    OLLAMA_HOST: Optional[str] = None
    OLLAMA_MODEL: Optional[str] = None
    WATCHLIST: Optional[str] = None
    DECISION_INTERVAL_MINUTES: Optional[str] = None


@app.post("/api/settings")
async def api_save_settings(body: SettingsUpdate):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    # Don't overwrite masked placeholders
    env = _read_env()
    for k in MASKED_KEYS:
        if k in updates and updates[k].startswith("****"):
            updates.pop(k)
    env.update(updates)
    _write_env(env)

    # Reload config in this process
    import importlib
    import config as cfg
    importlib.reload(cfg)

    return {"saved": True}


# ── WebSocket: live logs ──────────────────────────────────────────────────────
@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _log_subscribers.append(q)

    # Send last 50 lines of existing log file first
    if TRADER_LOG.exists():
        try:
            lines = TRADER_LOG.read_text(errors="replace").splitlines()[-50:]
            for line in lines:
                await websocket.send_text(line)
        except Exception:
            pass

    try:
        while True:
            try:
                line = await asyncio.wait_for(q.get(), timeout=30)
                await websocket.send_text(line)
            except asyncio.TimeoutError:
                await websocket.send_text("__ping__")
    except WebSocketDisconnect:
        pass
    finally:
        _log_subscribers.remove(q)
