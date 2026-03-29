"""
Fetches historical bars and current quotes from Alpaca.
"""
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
from loguru import logger
import config


def _timeframe():
    tf = config.BAR_TIMEFRAME
    mapping = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    return mapping.get(tf, TimeFrame(15, TimeFrameUnit.Minute))


def get_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Return a dict of symbol -> DataFrame with OHLCV bars."""
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    end = datetime.now(timezone.utc)
    # pull extra history to ensure we have enough after market-hours gaps
    start = end - timedelta(days=7)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=_timeframe(),
        start=start,
        end=end,
        limit=config.BARS_LOOKBACK * 3,
    )

    try:
        bars = client.get_stock_bars(request)
        result = {}
        for symbol in symbols:
            try:
                df = bars[symbol].df
                df = df.sort_index().tail(config.BARS_LOOKBACK)
                result[symbol] = df
            except (KeyError, Exception) as e:
                logger.warning(f"No bar data for {symbol}: {e}")
        return result
    except Exception as e:
        logger.error(f"Failed to fetch bars: {e}")
        return {}


def get_latest_quotes(symbols: list[str]) -> dict[str, dict]:
    """Return latest bid/ask/price for each symbol."""
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
    try:
        quotes = client.get_stock_latest_quote(request)
        return {
            sym: {
                "bid": float(q.bid_price),
                "ask": float(q.ask_price),
                "mid": round((float(q.bid_price) + float(q.ask_price)) / 2, 4),
            }
            for sym, q in quotes.items()
        }
    except Exception as e:
        logger.error(f"Failed to fetch quotes: {e}")
        return {}
