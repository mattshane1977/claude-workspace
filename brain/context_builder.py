"""
Assembles the full market snapshot that gets sent to the LLM.
"""
import json
from loguru import logger
from data import market_feed, indicators
from risk import portfolio_state
import config


def build(symbols: list[str]) -> dict:
    """
    Returns a structured context dict containing:
    - portfolio state
    - per-symbol price data + indicators
    - watchlist
    """
    logger.info(f"Building context for {symbols}")

    state = portfolio_state.get_state()
    bars = market_feed.get_bars(symbols)
    quotes = market_feed.get_latest_quotes(symbols)

    market_data = {}
    for symbol in symbols:
        df = bars.get(symbol)
        ind = indicators.compute(df) if df is not None and not df.empty else {}
        quote = quotes.get(symbol, {})

        # Recent bar history (last 10 bars as simple list for LLM)
        bar_history = []
        if df is not None and not df.empty:
            for _, row in df.tail(10).iterrows():
                bar_history.append({
                    "close": round(float(row["close"]), 4),
                    "volume": int(row["volume"]),
                })

        market_data[symbol] = {
            "quote": quote,
            "indicators": ind,
            "recent_bars": bar_history,
        }

    return {
        "portfolio": state,
        "market": market_data,
        "watchlist": symbols,
        "risk_rules": config.RISK,
    }


def to_prompt_text(context: dict) -> str:
    """Serialize context to a compact JSON string for prompt injection."""
    return json.dumps(context, indent=2)
