"""
High-level order execution that combines guardrails + alpaca_client.
"""
from loguru import logger
from risk import guardrails, portfolio_state
from execution import alpaca_client


def execute(actions: list[dict]) -> list[dict]:
    """
    Given a list of validated action dicts from decision_parser,
    run each through guardrails and execute approved orders.
    Returns list of execution results.
    """
    results = []
    state = portfolio_state.get_state()

    if not state:
        logger.error("Cannot execute: failed to fetch portfolio state")
        return []

    for action in actions:
        symbol = action["symbol"]
        act = action["action"]
        qty = action["qty"]

        if act == "hold":
            logger.info(f"  [HOLD] {symbol} — {action.get('reason', '')}")
            results.append({"symbol": symbol, "action": "hold", "executed": False})
            continue

        # Run through guardrails
        approved, reason = guardrails.approve(action, state)
        if not approved:
            logger.warning(f"  [BLOCKED] {act.upper()} {symbol} x{qty}: {reason}")
            results.append({
                "symbol": symbol,
                "action": act,
                "qty": qty,
                "executed": False,
                "blocked_reason": reason,
            })
            continue

        # Execute
        order = alpaca_client.place_market_order(symbol, qty, act)
        if order:
            results.append({
                "symbol": symbol,
                "action": act,
                "qty": qty,
                "executed": True,
                "order_id": order["id"],
            })
        else:
            results.append({
                "symbol": symbol,
                "action": act,
                "qty": qty,
                "executed": False,
                "blocked_reason": "order submission failed",
            })

    return results
