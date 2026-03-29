"""
Autonomous AI Stock Trader
Entry point — runs the decision loop on a schedule.

Usage:
    python main.py

Environment:
    Copy .env.example to .env and fill in your credentials.
"""
import sys
import signal
from datetime import datetime
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler

import config
import db
from brain import context_builder, llm_agent, decision_parser
from execution import alpaca_client, order_manager


# ── Logging setup ────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/trader.log", rotation="1 day", retention="30 days",
           level="DEBUG", encoding="utf-8")


# ── Graceful shutdown ─────────────────────────────────────────────────────────
def _shutdown(sig, frame):
    logger.warning("Shutdown signal received — cancelling open orders...")
    alpaca_client.cancel_all_orders()
    logger.info("Goodbye.")
    sys.exit(0)

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Core decision cycle ───────────────────────────────────────────────────────
def run_cycle():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"═══ Decision cycle starting @ {now} ═══")

    # 1. Check market is open
    if not alpaca_client.is_market_open():
        logger.info("Market is closed. Skipping cycle.")
        return

    # 2. Build context
    context = context_builder.build(config.WATCHLIST)
    context_text = context_builder.to_prompt_text(context)

    portfolio = context.get("portfolio", {})
    logger.info(
        f"Portfolio: equity=${portfolio.get('equity', 0):.2f}  "
        f"cash=${portfolio.get('cash', 0):.2f}  "
        f"daily_pnl={portfolio.get('daily_pnl_pct', 0):+.2f}%  "
        f"positions={portfolio.get('num_positions', 0)}"
    )

    # 3. Ask the LLM
    raw_decision = llm_agent.decide(context_text)
    if not raw_decision:
        logger.warning("No decision returned from LLM. Skipping execution.")
        return

    summary = raw_decision.get("summary", "")

    # 4. Parse + validate LLM output
    actions = decision_parser.parse(raw_decision)
    if not actions:
        logger.warning("No valid actions after parsing. Skipping execution.")
        return

    # 5. Persist cycle + decisions to DB
    cycle_id = db.insert_cycle(portfolio, summary)
    db.insert_decisions(cycle_id, actions)

    # 6. Execute through guardrails
    results = order_manager.execute(actions)
    db.insert_trades(cycle_id, results)

    executed = [r for r in results if r.get("executed")]
    blocked = [r for r in results if not r.get("executed") and r.get("action") != "hold"]

    logger.info(
        f"Cycle complete: {len(executed)} orders executed, "
        f"{len(blocked)} blocked, "
        f"{len(actions) - len(executed) - len(blocked)} held"
    )
    logger.info("═" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    db.init()

    logger.info("Autonomous AI Trader starting up")
    logger.info(f"   Watchlist: {config.WATCHLIST}")
    logger.info(f"   Model: {config.OLLAMA_MODEL}")
    logger.info(f"   Decision interval: every {config.DECISION_INTERVAL_MINUTES} minutes")
    logger.info(f"   Risk rules: {config.RISK}")

    # Run once immediately on startup
    run_cycle()

    # Then on schedule
    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        run_cycle,
        trigger="cron",
        # Run every N minutes during market hours Mon-Fri
        day_of_week="mon-fri",
        hour="9-15",
        minute=f"*/{config.DECISION_INTERVAL_MINUTES}",
    )

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    scheduler.start()


if __name__ == "__main__":
    main()
