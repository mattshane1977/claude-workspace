"""
Hard risk rules. Every LLM decision passes through here before execution.
If any rule fails, the order is blocked and logged.
"""
from loguru import logger
import config
from risk import portfolio_state


# Track peak equity in-process (resets each run, use DB for persistence)
_peak_equity: float = 0.0


def approve(decision: dict, state: dict) -> tuple[bool, str]:
    """
    Returns (approved: bool, reason: str).
    decision keys: symbol, action, qty, price_estimate
    state: output of portfolio_state.get_state()
    """
    global _peak_equity

    symbol = decision.get("symbol", "?")
    action = decision.get("action", "hold")
    qty = decision.get("qty", 0)

    if action == "hold":
        return True, "hold approved"

    equity = state.get("equity", 0)
    cash = state.get("cash", 0)
    daily_pnl_pct = state.get("daily_pnl_pct", 0)

    # Update peak equity tracker
    if equity > _peak_equity:
        _peak_equity = equity

    # 1. Daily loss circuit breaker
    if daily_pnl_pct < -abs(config.RISK["max_daily_loss_pct"] * 100):
        reason = f"BLOCKED: daily loss {daily_pnl_pct:.2f}% exceeds limit"
        logger.warning(reason)
        return False, reason

    # 2. Max drawdown from peak
    if _peak_equity > 0:
        drawdown_pct = (_peak_equity - equity) / _peak_equity
        if drawdown_pct > config.RISK["max_drawdown_pct"]:
            reason = f"BLOCKED: drawdown {drawdown_pct*100:.2f}% exceeds {config.RISK['max_drawdown_pct']*100:.0f}% limit"
            logger.warning(reason)
            return False, reason

    # 3. Max trades per day
    trades_today = portfolio_state.count_trades_today()
    if trades_today >= config.RISK["max_trades_per_day"]:
        reason = f"BLOCKED: {trades_today} trades today, daily limit reached"
        logger.warning(reason)
        return False, reason

    # 4. Qty hard cap
    if qty > config.RISK["max_qty_per_order"]:
        reason = f"BLOCKED: qty {qty} exceeds max_qty_per_order {config.RISK['max_qty_per_order']}"
        logger.warning(reason)
        return False, reason

    if action == "buy":
        price_est = decision.get("price_estimate", 0)
        order_value = price_est * qty if price_est else 0

        # 5. Sufficient cash
        min_cash = equity * config.RISK["min_cash_reserve_pct"]
        if order_value > 0 and (cash - order_value) < min_cash:
            reason = f"BLOCKED: insufficient cash. Have ${cash:.2f}, need ${order_value:.2f} + ${min_cash:.2f} reserve"
            logger.warning(reason)
            return False, reason

        # 6. Position size limit
        if order_value > 0:
            max_position_value = equity * config.RISK["max_position_pct"]
            # Add existing position value if any
            existing = next(
                (p for p in state.get("positions", []) if p["symbol"] == symbol), None
            )
            existing_value = existing["market_value"] if existing else 0
            if existing_value + order_value > max_position_value:
                reason = f"BLOCKED: {symbol} position would exceed {config.RISK['max_position_pct']*100:.0f}% of portfolio"
                logger.warning(reason)
                return False, reason

    if action == "sell":
        # 7. Must actually hold the position
        existing = next(
            (p for p in state.get("positions", []) if p["symbol"] == symbol), None
        )
        if not existing:
            reason = f"BLOCKED: sell {symbol} but no position held"
            logger.warning(reason)
            return False, reason
        if qty > existing["qty"]:
            reason = f"BLOCKED: sell qty {qty} > held qty {existing['qty']}"
            logger.warning(reason)
            return False, reason

    return True, "approved"
