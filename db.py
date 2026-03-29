"""
SQLite persistence layer — stores decision cycles, LLM decisions, and trades.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

DB_PATH = Path(__file__).parent / "data" / "trader.db"


def _conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cycles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                equity      REAL,
                cash        REAL,
                cash_pct    REAL,
                daily_pnl_pct REAL,
                num_positions INTEGER,
                summary     TEXT
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id    INTEGER REFERENCES cycles(id),
                ts          TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                action      TEXT NOT NULL,
                qty         INTEGER,
                price_estimate REAL,
                confidence  REAL,
                reason      TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id    INTEGER REFERENCES cycles(id),
                ts          TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                action      TEXT NOT NULL,
                qty         INTEGER,
                order_id    TEXT,
                executed    INTEGER NOT NULL DEFAULT 0,
                blocked_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS research (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT NOT NULL,
                symbol        TEXT NOT NULL,
                company       TEXT,
                sector        TEXT,
                price         REAL,
                market_cap    REAL,
                pe_ratio      REAL,
                week52_high   REAL,
                week52_low    REAL,
                volume_vs_avg REAL,
                ai_score      REAL,
                ai_verdict    TEXT,
                ai_reason     TEXT,
                news_snippet  TEXT,
                run_summary   TEXT
            );

            CREATE TABLE IF NOT EXISTS ipo_watch (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT NOT NULL,
                company         TEXT,
                ticker          TEXT,
                cik             TEXT,
                filed_date      TEXT,
                form_type       TEXT,
                sic             TEXT,
                location        TEXT,
                description     TEXT,
                space_keywords  TEXT,
                ai_score        REAL,
                ai_verdict      TEXT,
                ai_reason       TEXT,
                edgar_url       TEXT,
                llm_summary     TEXT
            );

            CREATE TABLE IF NOT EXISTS backtest_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                symbols     TEXT,
                days        INTEGER,
                result_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_ipo_watch_ts ON ipo_watch(ts);
            CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
            CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
            CREATE INDEX IF NOT EXISTS idx_cycles_ts ON cycles(ts);
            CREATE INDEX IF NOT EXISTS idx_research_ts ON research(ts);
        """)
    # Add signals column if it doesn't exist yet (migration for existing DBs)
    try:
        with _conn() as conn:
            conn.execute("ALTER TABLE research ADD COLUMN signals TEXT")
    except Exception:
        pass  # column already exists

    logger.info(f"DB initialised at {DB_PATH}")


def insert_cycle(portfolio: dict, summary: str = "") -> int:
    ts = datetime.utcnow().isoformat()
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO cycles (ts, equity, cash, cash_pct, daily_pnl_pct, num_positions, summary)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ts,
                portfolio.get("equity"),
                portfolio.get("cash"),
                portfolio.get("cash_pct"),
                portfolio.get("daily_pnl_pct"),
                portfolio.get("num_positions", 0),
                summary,
            ),
        )
        return cur.lastrowid


def insert_decisions(cycle_id: int, actions: list[dict]):
    ts = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.executemany(
            """INSERT INTO decisions (cycle_id, ts, symbol, action, qty, price_estimate, confidence, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    cycle_id,
                    ts,
                    a["symbol"],
                    a["action"],
                    a.get("qty", 0),
                    a.get("price_estimate"),
                    a.get("confidence"),
                    a.get("reason", ""),
                )
                for a in actions
            ],
        )


def insert_trades(cycle_id: int, results: list[dict]):
    ts = datetime.utcnow().isoformat()
    rows = [
        r for r in results
        if r.get("action") in ("buy", "sell")
    ]
    if not rows:
        return
    with _conn() as conn:
        conn.executemany(
            """INSERT INTO trades (cycle_id, ts, symbol, action, qty, order_id, executed, blocked_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    cycle_id,
                    ts,
                    r["symbol"],
                    r["action"],
                    r.get("qty", 0),
                    r.get("order_id"),
                    1 if r.get("executed") else 0,
                    r.get("blocked_reason"),
                )
                for r in rows
            ],
        )


def get_cycles(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cycles ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_decisions(limit: int = 100, symbol: str = None) -> list[dict]:
    with _conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE symbol=? ORDER BY id DESC LIMIT ?",
                (symbol.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_trades(limit: int = 100, symbol: str = None) -> list[dict]:
    with _conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ?",
                (symbol.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def insert_ipo_watch(results: list[dict]):
    ts = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.executemany(
            """INSERT INTO ipo_watch
               (ts, company, ticker, cik, filed_date, form_type, sic, location,
                description, space_keywords, ai_score, ai_verdict, ai_reason, edgar_url, llm_summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    ts,
                    r.get("company"), r.get("ticker"), r.get("cik"),
                    r.get("filed_date"), r.get("form_type"), r.get("sic"),
                    r.get("location"), r.get("description"), r.get("space_keywords"),
                    r.get("ai_score"), r.get("ai_verdict"), r.get("ai_reason"),
                    r.get("edgar_url"), r.get("llm_summary"),
                )
                for r in results
            ],
        )


def get_ipo_watch(limit: int = 200) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ipo_watch ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def insert_research(results: list[dict]):
    ts = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.executemany(
            """INSERT INTO research
               (ts, symbol, company, sector, price, market_cap, pe_ratio,
                week52_high, week52_low, volume_vs_avg,
                ai_score, ai_verdict, ai_reason, news_snippet, run_summary, signals)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    ts,
                    r["symbol"],
                    r.get("company"),
                    r.get("sector"),
                    r.get("price"),
                    r.get("market_cap"),
                    r.get("pe_ratio"),
                    r.get("week52_high"),
                    r.get("week52_low"),
                    r.get("volume_vs_avg"),
                    r.get("ai_score"),
                    r.get("ai_verdict"),
                    r.get("ai_reason"),
                    r.get("news_snippet"),
                    r.get("llm_summary"),
                    r.get("signals"),
                )
                for r in results
            ],
        )


def get_research(limit: int = 200) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM research ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def insert_backtest(result: dict):
    ts = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO backtest_runs (ts, symbols, days, result_json) VALUES (?, ?, ?, ?)",
            (ts, ",".join(result.get("symbols", [])), result.get("days"), json.dumps(result)),
        )


def get_backtests(limit: int = 10) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, ts, symbols, days, result_json FROM backtest_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["result"] = json.loads(d.pop("result_json"))
        except Exception:
            d["result"] = {}
        out.append(d)
    return out


def get_equity_history(limit: int = 200) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, equity, cash, daily_pnl_pct FROM cycles ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return list(reversed([dict(r) for r in rows]))
