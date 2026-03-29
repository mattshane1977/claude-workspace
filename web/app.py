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
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
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
PROMOTE_THRESHOLD = 0.80   # AI score needed to auto-add to watchlist
PROMOTE_MAX = 3            # max new symbols added per scout run


def _auto_promote(results: list[dict], source: str = "Scout"):
    """
    Add high-scoring gems to the watchlist automatically.
    Only adds stocks with a valid ticker and score >= PROMOTE_THRESHOLD.
    """
    import importlib
    import config as cfg
    importlib.reload(cfg)

    current = {s.upper() for s in cfg.WATCHLIST}
    to_add = []

    for r in sorted(results, key=lambda x: x.get("ai_score") or 0, reverse=True):
        if len(to_add) >= PROMOTE_MAX:
            break
        symbol = (r.get("symbol") or r.get("ticker") or "").strip().upper()
        score = r.get("ai_score") or 0
        if not symbol or not symbol.isalpha() or len(symbol) > 5:
            continue
        if symbol in current or symbol in to_add:
            continue
        if score < PROMOTE_THRESHOLD:
            break  # sorted by score, so nothing below threshold after this
        to_add.append(symbol)

    if not to_add:
        return

    new_watchlist = list(cfg.WATCHLIST) + to_add
    env = _read_env()
    env["WATCHLIST"] = ",".join(new_watchlist)
    _write_env(env)

    import importlib
    import config as cfg2
    importlib.reload(cfg2)

    for sym in to_add:
        score = next((r.get("ai_score", 0) for r in results if (r.get("symbol") or r.get("ticker", "")).upper() == sym), 0)
        _broadcast_log(f"[Auto-Promote] {sym} added to watchlist from {source} (score {score:.0%})")

    _broadcast_log(f"[Auto-Promote] Watchlist is now: {','.join(new_watchlist)}")


def _scheduled_eod_summary():
    """Broadcast an end-of-day performance summary at 4:15 PM ET weekdays."""
    import importlib
    import config as cfg
    importlib.reload(cfg)
    try:
        from risk.portfolio_state import get_state
        state = get_state()
        equity = state.get("equity") or 0
        cash = state.get("cash") or 0
        pnl = state.get("daily_pnl_pct") or 0
        positions = state.get("positions") or []
        trades_today = db.get_trades(limit=200)

        sep = "=" * 56
        _broadcast_log(sep)
        _broadcast_log(f"[EOD Summary] {datetime.now().strftime('%A, %B %d %Y')}")
        _broadcast_log(f"[EOD Summary] Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}  |  Day P&L: {pnl:+.2f}%")
        _broadcast_log(f"[EOD Summary] Open positions: {len(positions)}")
        for p in positions[:10]:
            upnl = p.get("unrealized_pnl_pct") or 0
            _broadcast_log(
                f"[EOD Summary]   {p['symbol']:6s}  {p.get('qty')} shares @ {p.get('avg_entry', 0):.2f}"
                f"  unrealized {upnl:+.1f}%"
            )
        executed_ct = sum(1 for t in trades_today if t.get("executed"))
        _broadcast_log(f"[EOD Summary] Trades executed today: {executed_ct}")
        _broadcast_log(sep)
    except Exception as e:
        _broadcast_log(f"[EOD Summary] Error generating summary: {e}")
        logger.error(f"EOD summary error: {e}")


def _scheduled_ipo_scout():
    """Runs the IPO/space-supplier scout on schedule."""
    import importlib
    import config as cfg
    importlib.reload(cfg)
    from brain.ipo_scout import run_ipo_scout
    try:
        _broadcast_log("Scheduled IPO scout starting (7:00 AM ET daily run)...")
        results = run_ipo_scout(s1_days=30, keyword_days=30)
        if results:
            db.insert_ipo_watch(results)
            hot = sum(1 for r in results if r.get("ai_verdict") == "Hot")
            _broadcast_log(f"IPO scout complete: {len(results)} companies rated, {hot} marked Hot.")
            _auto_promote(results, source="IPO Scout")
        else:
            _broadcast_log("IPO scout complete: no results found.")
    except Exception as e:
        _broadcast_log(f"IPO scout error: {e}")
        logger.error(f"IPO scout error: {e}")


def _scheduled_scout():
    """Runs the stock scout on schedule and saves results to DB."""
    import importlib
    import config as cfg
    importlib.reload(cfg)
    from brain.stock_scout import run_scout
    try:
        _broadcast_log("Scheduled scout starting (9:35 AM ET daily run)...")
        results = run_scout(cfg.WATCHLIST)
        if results:
            db.insert_research(results)
            _broadcast_log(f"Scheduled scout complete: {len(results)} stocks analyzed and saved.")
            _auto_promote(results, source="Scout")
        else:
            _broadcast_log("Scheduled scout complete: no results found.")
    except Exception as e:
        _broadcast_log(f"Scheduled scout error: {e}")
        logger.error(f"Scheduled scout error: {e}")


_scheduler = BackgroundScheduler(timezone="America/New_York")
_scheduler.add_job(
    _scheduled_ipo_scout,
    trigger="cron",
    day_of_week="mon-fri",
    hour=7,
    minute=0,
    id="daily_ipo_scout",
)
_scheduler.add_job(
    _scheduled_scout,
    trigger="cron",
    day_of_week="mon-fri",
    hour=9,
    minute=35,
    id="daily_scout",
)
_scheduler.add_job(
    _scheduled_eod_summary,
    trigger="cron",
    day_of_week="mon-fri",
    hour=16,
    minute=15,
    id="eod_summary",
)


@app.on_event("startup")
async def on_startup():
    db.init()
    # Wire LLM streaming to the dashboard log
    from brain import llm_stream
    llm_stream.set_broadcast(_broadcast_log)
    _scheduler.start()
    logger.info("Web server started. Trader not auto-started — use the UI.")
    logger.info("Stock scout scheduled: weekdays at 9:35 AM ET")
    logger.info("IPO scout scheduled: weekdays at 7:00 AM ET")


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


@app.get("/api/stock-detail/{symbol}")
async def api_stock_detail(symbol: str):
    try:
        import yfinance as yf
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        hist = ticker.history(period="6mo")

        history = []
        for date, row in hist.iterrows():
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        return {
            "symbol": symbol,
            "company": info.get("shortName", symbol),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_pct": info.get("52WeekChange"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "week52_high": info.get("fiftyTwoWeekHigh"),
            "week52_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "dividend_yield": info.get("dividendYield"),
            "description": (info.get("longBusinessSummary") or "")[:600],
            "history": history,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_prev_close_cache: dict = {}   # symbol -> {prev_close, cached_at}
_PREV_CLOSE_TTL = 300          # refresh at most every 5 min


def _get_prev_close(symbol: str) -> float | None:
    import yfinance as yf
    cached = _prev_close_cache.get(symbol, {})
    if time.time() - cached.get("cached_at", 0) < _PREV_CLOSE_TTL:
        return cached.get("prev_close")
    try:
        fi = yf.Ticker(symbol).fast_info
        pc = float(fi.previous_close)
        _prev_close_cache[symbol] = {"prev_close": pc, "cached_at": time.time()}
        return pc
    except Exception:
        return None


@app.get("/api/stock-search")
async def api_stock_search(q: str):
    try:
        import yfinance as yf
        search = yf.Search(q, max_results=12)
        quotes = getattr(search, 'quotes', []) or []
        results = []
        for item in quotes:
            symbol = item.get('symbol', '')
            if not symbol:
                continue
            quote_type = item.get('quoteType', '')
            if quote_type not in ('EQUITY', 'ETF', ''):
                continue
            results.append({
                'symbol': symbol,
                'company': item.get('longname') or item.get('shortname', symbol),
                'exchange': item.get('exchange', ''),
                'type': quote_type,
            })
        return results[:8]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/watchlist-prices")
async def api_watchlist_prices():
    try:
        import importlib
        import config as cfg
        importlib.reload(cfg)
        from data.market_feed import get_latest_quotes
        quotes = get_latest_quotes(cfg.WATCHLIST)
        result = []
        for s in cfg.WATCHLIST:
            q = quotes.get(s, {"bid": None, "ask": None, "mid": None})
            pc = _get_prev_close(s)
            mid = q.get("mid")
            raw_chg = (mid - pc) / pc * 100 if mid and pc else None
            change_pct = round(raw_chg, 2) if raw_chg is not None and abs(raw_chg) < 25 else None
            result.append({"symbol": s, **q, "prev_close": pc, "change_pct": change_pct})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research")
async def api_research(limit: int = 200):
    return db.get_research(limit)


@app.get("/api/ipo-watch")
async def api_ipo_watch(limit: int = 200):
    return db.get_ipo_watch(limit)


@app.post("/api/ipo-watch/run")
async def api_run_ipo_scout():
    """Trigger a background IPO/space-supplier scouting run."""
    import threading

    def _run():
        from brain.ipo_scout import run_ipo_scout
        try:
            _broadcast_log("IPO scout started — scanning EDGAR for S-1 filings and space/defense suppliers...")
            results = run_ipo_scout(s1_days=30, keyword_days=30)
            if results:
                db.insert_ipo_watch(results)
                hot = sum(1 for r in results if r.get("ai_verdict") == "Hot")
                _broadcast_log(f"IPO scout complete: {len(results)} companies rated, {hot} marked Hot.")
                _auto_promote(results, source="IPO Scout")
            else:
                _broadcast_log("IPO scout complete: no results found.")
        except Exception as e:
            _broadcast_log(f"IPO scout error: {e}")
            logger.error(f"IPO scout error: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return {"queued": True}


class WatchlistAdd(BaseModel):
    symbol: str


@app.post("/api/watchlist/remove")
async def api_watchlist_remove(body: WatchlistAdd):
    symbol = body.symbol.strip().upper()
    import importlib
    import config as cfg
    importlib.reload(cfg)
    current_upper = [s.upper() for s in cfg.WATCHLIST]
    if symbol not in current_upper:
        return {"removed": False, "reason": "not on watchlist"}
    new_watchlist = [s for s in cfg.WATCHLIST if s.upper() != symbol]
    env = _read_env()
    env["WATCHLIST"] = ",".join(new_watchlist)
    _write_env(env)
    importlib.reload(cfg)
    _broadcast_log(f"[Watchlist] {symbol} removed. Watchlist: {','.join(new_watchlist)}")
    return {"removed": True, "symbol": symbol, "watchlist": new_watchlist}


@app.post("/api/watchlist/add")
async def api_watchlist_add(body: WatchlistAdd):
    symbol = body.symbol.strip().upper()
    if not symbol or not symbol.isalpha() or len(symbol) > 5:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    import importlib
    import config as cfg
    importlib.reload(cfg)
    if symbol in [s.upper() for s in cfg.WATCHLIST]:
        return {"added": False, "reason": "already on watchlist", "watchlist": cfg.WATCHLIST}
    new_watchlist = list(cfg.WATCHLIST) + [symbol]
    env = _read_env()
    env["WATCHLIST"] = ",".join(new_watchlist)
    _write_env(env)
    importlib.reload(cfg)
    _broadcast_log(f"[Watchlist] {symbol} manually added. Watchlist: {','.join(new_watchlist)}")
    return {"added": True, "symbol": symbol, "watchlist": new_watchlist}


@app.get("/api/news")
async def api_news():
    try:
        import importlib
        import config as cfg
        import yfinance as yf
        importlib.reload(cfg)
        all_news = []
        for symbol in cfg.WATCHLIST:
            try:
                ticker = yf.Ticker(symbol)
                for n in (ticker.news or [])[:5]:
                    content = n.get("content", {}) if isinstance(n.get("content"), dict) else {}
                    title = content.get("title") or n.get("title", "")
                    if not title:
                        continue
                    pub_date = content.get("pubDate") or n.get("providerPublishTime", "")
                    canon = content.get("canonicalUrl", {})
                    url = canon.get("url", "") if isinstance(canon, dict) else ""
                    provider = content.get("provider", {})
                    publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
                    all_news.append({
                        "symbol": symbol,
                        "title": title,
                        "publisher": publisher,
                        "url": url,
                        "pub_date": str(pub_date),
                    })
            except Exception:
                continue
        all_news.sort(key=lambda x: x.get("pub_date") or "", reverse=True)
        return all_news[:100]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance")
async def api_performance():
    try:
        trades = db.get_trades(limit=10000)
        history = db.get_equity_history(limit=10000)

        executed = [t for t in trades if t.get("executed")]
        total_buys = sum(1 for t in executed if t["action"] == "buy")
        total_sells = sum(1 for t in executed if t["action"] == "sell")

        equities = [c["equity"] for c in history if c.get("equity")]
        start_eq = equities[0] if equities else None
        end_eq = equities[-1] if equities else None
        total_return = (end_eq - start_eq) / start_eq * 100 if start_eq and end_eq else None

        max_eq, max_dd = 0.0, 0.0
        for eq in equities:
            if eq > max_eq:
                max_eq = eq
            if max_eq > 0:
                dd = (max_eq - eq) / max_eq * 100
                if dd > max_dd:
                    max_dd = dd

        cycle_wins = sum(
            1 for i in range(1, len(equities))
            if equities[i] is not None and equities[i - 1] is not None and equities[i] > equities[i - 1]
        )
        cycle_total = max(len(equities) - 1, 0)
        cycle_win_rate = cycle_wins / cycle_total * 100 if cycle_total else None

        return {
            "total_executed_trades": len(executed),
            "total_buys": total_buys,
            "total_sells": total_sells,
            "total_return_pct": round(total_return, 2) if total_return is not None else None,
            "max_drawdown_pct": round(max_dd, 2),
            "start_equity": start_eq,
            "end_equity": end_eq,
            "cycle_win_rate": round(cycle_win_rate, 1) if cycle_win_rate is not None else None,
            "num_cycles": len(history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_sector_cache: dict = {}
_SECTOR_CACHE_TTL = 1800   # 30 min


@app.get("/api/sector-exposure")
async def api_sector_exposure():
    try:
        import yfinance as yf
        from risk.portfolio_state import get_state
        positions = get_state().get("positions") or []
        if not positions:
            return []

        result = []
        for pos in positions:
            sym = pos["symbol"]
            cached = _sector_cache.get(sym, {})
            if time.time() - cached.get("cached_at", 0) < _SECTOR_CACHE_TTL:
                sector = cached["sector"]
            else:
                try:
                    sector = yf.Ticker(sym).info.get("sector", "Unknown") or "Unknown"
                    _sector_cache[sym] = {"sector": sector, "cached_at": time.time()}
                except Exception:
                    sector = "Unknown"
            result.append({
                "symbol": sym,
                "sector": sector,
                "market_value": pos.get("market_value", 0),
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/run")
async def api_run_backtest(body: dict = None):
    import threading
    import importlib
    import config as cfg
    importlib.reload(cfg)

    params = body or {}
    symbols = [s.strip().upper() for s in params.get("symbols", cfg.WATCHLIST) if s.strip()]
    days = int(params.get("days", 90))
    starting_equity = float(params.get("starting_equity", 100_000))

    def _run():
        from brain.backtest import run_backtest
        try:
            _broadcast_log(f"Backtest starting: {symbols}, {days} days, ${starting_equity:,.0f} capital...")
            result = run_backtest(symbols, days=days, starting_equity=starting_equity)
            if "error" in result:
                _broadcast_log(f"Backtest error: {result['error']}")
                return
            db.insert_backtest(result)
            _broadcast_log(
                f"Backtest complete: {result['total_return_pct']:+.1f}% return vs "
                f"buy-and-hold {result['bnh_return_pct']:+.1f}% | "
                f"{result['total_trades']} trades | max drawdown {result['max_drawdown_pct']:.1f}%"
            )
        except Exception as e:
            _broadcast_log(f"Backtest error: {e}")
            logger.error(f"Backtest error: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return {"queued": True}


@app.get("/api/backtest/results")
async def api_backtest_results():
    return db.get_backtests(limit=5)


@app.get("/api/ipo-watch/schedule")
async def api_ipo_schedule():
    job = _scheduler.get_job("daily_ipo_scout")
    if job and job.next_run_time:
        return {"next_run": job.next_run_time.isoformat(), "schedule": "Weekdays 7:00 AM ET"}
    return {"next_run": None, "schedule": "Weekdays 7:00 AM ET"}


@app.get("/api/scout/schedule")
async def api_scout_schedule():
    job = _scheduler.get_job("daily_scout")
    if job and job.next_run_time:
        return {"next_run": job.next_run_time.isoformat(), "schedule": "Weekdays 9:35 AM ET"}
    return {"next_run": None, "schedule": "Weekdays 9:35 AM ET"}


@app.post("/api/scout/run")
async def api_run_scout():
    """Trigger a background stock scouting run."""
    import threading

    def _run():
        import importlib
        import config as cfg
        importlib.reload(cfg)
        from brain.stock_scout import run_scout
        try:
            _broadcast_log("Scout started — fetching trending stocks from Yahoo Finance...")
            results = run_scout(cfg.WATCHLIST)
            if results:
                db.insert_research(results)
                _broadcast_log(f"Scout complete: {len(results)} stocks analyzed and saved.")
                _auto_promote(results, source="Scout")
            else:
                _broadcast_log("Scout complete: no results found.")
        except Exception as e:
            _broadcast_log(f"Scout error: {e}")
            logger.error(f"Scout error: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"queued": True}


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
