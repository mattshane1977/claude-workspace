"""
Compute technical indicators on a price DataFrame using pandas-ta.
Returns a clean dict the LLM can read.
"""
import pandas as pd
import pandas_ta as ta
from loguru import logger


def compute(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame with columns [open, high, low, close, volume],
    return a dict of indicator values (most recent bar).
    """
    if df is None or len(df) < 5:
        return {}

    try:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        result = {}

        # Price summary
        result["price_current"] = round(float(close.iloc[-1]), 4)
        result["price_open"] = round(float(df["open"].iloc[-1]), 4)
        result["price_high_session"] = round(float(high.iloc[-1]), 4)
        result["price_low_session"] = round(float(low.iloc[-1]), 4)
        result["volume"] = int(volume.iloc[-1])
        result["avg_volume_10"] = int(volume.tail(10).mean())

        # EMAs
        ema9 = ta.ema(close, length=9)
        ema20 = ta.ema(close, length=20)
        if ema9 is not None and not ema9.empty:
            result["ema9"] = round(float(ema9.iloc[-1]), 4)
        if ema20 is not None and not ema20.empty:
            result["ema20"] = round(float(ema20.iloc[-1]), 4)

        # RSI
        rsi = ta.rsi(close, length=14)
        if rsi is not None and not rsi.empty:
            result["rsi14"] = round(float(rsi.iloc[-1]), 2)

        # MACD
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            result["macd"] = round(float(macd["MACD_12_26_9"].iloc[-1]), 4)
            result["macd_signal"] = round(float(macd["MACDs_12_26_9"].iloc[-1]), 4)
            result["macd_hist"] = round(float(macd["MACDh_12_26_9"].iloc[-1]), 4)

        # Bollinger Bands
        bbands = ta.bbands(close, length=20, std=2)
        if bbands is not None and not bbands.empty:
            result["bb_upper"] = round(float(bbands["BBU_20_2.0"].iloc[-1]), 4)
            result["bb_mid"] = round(float(bbands["BBM_20_2.0"].iloc[-1]), 4)
            result["bb_lower"] = round(float(bbands["BBL_20_2.0"].iloc[-1]), 4)
            bb_width = result["bb_upper"] - result["bb_lower"]
            result["bb_pct"] = round(
                (result["price_current"] - result["bb_lower"]) / bb_width, 4
            ) if bb_width > 0 else 0.5

        # ATR (volatility)
        atr = ta.atr(high, low, close, length=14)
        if atr is not None and not atr.empty:
            result["atr14"] = round(float(atr.iloc[-1]), 4)

        # Price change %
        if len(close) >= 2:
            result["change_pct_last_bar"] = round(
                (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2]) * 100, 3
            )
        if len(close) >= config_lookback(df):
            result["change_pct_session"] = round(
                (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0]) * 100, 3
            )

        # Trend: price vs EMAs
        if "ema9" in result and "ema20" in result:
            if result["price_current"] > result["ema9"] > result["ema20"]:
                result["trend"] = "bullish"
            elif result["price_current"] < result["ema9"] < result["ema20"]:
                result["trend"] = "bearish"
            else:
                result["trend"] = "mixed"

        return result

    except Exception as e:
        logger.error(f"Indicator computation failed: {e}")
        return {}


def config_lookback(df):
    return min(len(df), 30)
