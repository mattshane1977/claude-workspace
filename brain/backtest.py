"""
Fast rule-based backtester using yfinance daily OHLCV data.

Strategy: RSI(14) mean-reversion + EMA(20) trend filter
  BUY  when RSI < 40 AND price > EMA20  (oversold bounce in an uptrend)
  SELL when RSI > 70 OR price < EMA20 * 0.99  (overbought or trend break)

Max 5 concurrent positions, ≤25 % of equity per slot.
Also computes equal-weight buy-and-hold return for comparison.
"""
import pandas as pd
import yfinance as yf
from loguru import logger


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def run_backtest(symbols: list[str], days: int = 90,
                 starting_equity: float = 100_000.0) -> dict:
    logger.info(f"Backtest: {symbols}, {days}d, ${starting_equity:,.0f}")

    # Download OHLCV with extra warmup bars for indicators
    raw: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period=f"{days + 80}d")
            if hist.empty:
                continue
            hist.columns = [c.lower() for c in hist.columns]
            hist["rsi"] = _rsi(hist["close"])
            hist["ema20"] = hist["close"].ewm(span=20, adjust=False).mean()
            hist.index = hist.index.normalize()
            raw[sym] = hist
        except Exception as e:
            logger.warning(f"Backtest: no data for {sym}: {e}")

    if not raw:
        return {"error": "No historical data available for any symbol"}

    # Intersection of trading dates, trimmed to requested window
    common = sorted(set.intersection(*[set(df.index) for df in raw.values()]))
    sim_dates = common[-days:]
    if not sim_dates:
        return {"error": "Insufficient overlapping trading days"}

    # State
    cash = starting_equity
    positions: dict[str, dict] = {}  # sym -> {qty, cost_basis}
    equity_curve: list[dict] = []
    trades_log: list[dict] = []
    bnh_start: dict[str, float] = {}

    max_equity = starting_equity
    max_drawdown = 0.0
    wins = losses = 0
    MAX_POS = min(5, len(raw))

    # Buy-and-hold: record first-day open prices
    for sym, df in raw.items():
        rows = df[df.index == sim_dates[0]]
        if not rows.empty:
            bnh_start[sym] = float(rows.iloc[0]["open"])

    for date in sim_dates:
        day: dict[str, dict] = {}
        for sym, df in raw.items():
            rows = df[df.index == date]
            if rows.empty:
                continue
            r = rows.iloc[0]
            day[sym] = {
                "open": float(r["open"]),
                "close": float(r["close"]),
                "rsi": float(r["rsi"]) if not pd.isna(r["rsi"]) else 50.0,
                "ema20": float(r["ema20"]) if not pd.isna(r["ema20"]) else float(r["close"]),
            }

        # SELL signals
        to_sell = [
            s for s, pos in positions.items()
            if s in day and (day[s]["rsi"] > 70 or day[s]["close"] < day[s]["ema20"] * 0.99)
        ]
        for sym in to_sell:
            pos = positions.pop(sym)
            price = day[sym]["open"]
            proceeds = pos["qty"] * price
            cash += proceeds
            pnl = proceeds - pos["cost_basis"]
            if pnl >= 0:
                wins += 1
            else:
                losses += 1
            trades_log.append({
                "date": date.date().isoformat(), "symbol": sym,
                "action": "SELL", "qty": pos["qty"],
                "price": round(price, 2), "pnl": round(pnl, 2),
            })

        # BUY signals: RSI < 45 and price within 3% of EMA20 (allows slight dips below trend)
        slots = MAX_POS - len(positions)
        if slots > 0 and cash > 500:
            candidates = sorted(
                [(s, d) for s, d in day.items()
                 if s not in positions and d["rsi"] < 45 and d["close"] > d["ema20"] * 0.97],
                key=lambda x: x[1]["rsi"],
            )
            for sym, d in candidates[:slots]:
                alloc = min(cash * 0.25, cash / max(slots, 1))
                price = d["open"]
                if price <= 0:
                    continue
                qty = int(alloc / price)
                if qty <= 0:
                    continue
                cost = qty * price
                cash -= cost
                positions[sym] = {"qty": qty, "cost_basis": cost}
                trades_log.append({
                    "date": date.date().isoformat(), "symbol": sym,
                    "action": "BUY", "qty": qty,
                    "price": round(price, 2), "pnl": None,
                })

        # End-of-day mark-to-market
        eod = cash + sum(
            pos["qty"] * day[s]["close"]
            for s, pos in positions.items() if s in day
        )
        equity_curve.append({"date": date.date().isoformat(), "equity": round(eod, 2)})

        if eod > max_equity:
            max_equity = eod
        dd = (max_equity - eod) / max_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

    # Buy-and-hold final value
    bnh_equity = starting_equity
    if bnh_start:
        alloc_per = starting_equity / len(bnh_start)
        bnh_equity = 0.0
        last = sim_dates[-1]
        for sym, entry_price in bnh_start.items():
            rows = raw[sym][raw[sym].index == last]
            exit_price = float(rows.iloc[0]["close"]) if not rows.empty else entry_price
            bnh_equity += (alloc_per / entry_price) * exit_price

    total_trades = wins + losses
    ending = equity_curve[-1]["equity"] if equity_curve else starting_equity
    total_return = (ending - starting_equity) / starting_equity * 100
    bnh_return = (bnh_equity - starting_equity) / starting_equity * 100

    return {
        "symbols": symbols,
        "days": days,
        "starting_equity": starting_equity,
        "ending_equity": ending,
        "total_return_pct": round(total_return, 2),
        "bnh_return_pct": round(bnh_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total_trades * 100, 1) if total_trades else 0.0,
        "equity_curve": equity_curve,
        "trades": trades_log[-100:],
    }
