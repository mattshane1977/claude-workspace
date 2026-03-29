"""
Validates and normalizes LLM output before it hits the risk layer.
"""
from loguru import logger
import config


def parse(raw: dict) -> list[dict]:
    """
    Takes raw LLM output dict, returns a clean list of validated action dicts.
    Each action: {symbol, action, qty, price_estimate, confidence, reason}
    """
    if not raw or "actions" not in raw:
        logger.warning("LLM returned no 'actions' key")
        return []

    summary = raw.get("summary", "")
    if summary:
        logger.info(f"LLM summary: {summary}")

    valid_actions = []
    for item in raw["actions"]:
        try:
            symbol = str(item.get("symbol", "")).upper().strip()
            action = str(item.get("action", "hold")).lower().strip()
            qty = int(item.get("qty", 0))
            price_estimate = float(item.get("price_estimate", 0))
            confidence = float(item.get("confidence", 0.5))
            reason = str(item.get("reason", ""))

            # Validate symbol is on watchlist
            if symbol not in config.WATCHLIST:
                logger.warning(f"LLM returned symbol {symbol} not in watchlist, skipping")
                continue

            # Validate action
            if action not in ("buy", "sell", "hold"):
                logger.warning(f"Invalid action '{action}' for {symbol}, defaulting to hold")
                action = "hold"

            # Validate qty
            if qty < 0:
                logger.warning(f"Negative qty for {symbol}, setting to 0")
                qty = 0

            if action in ("buy", "sell") and qty == 0:
                logger.warning(f"{action} action for {symbol} has qty=0, converting to hold")
                action = "hold"

            # Low confidence threshold
            if confidence < 0.4 and action != "hold":
                logger.warning(f"Low confidence {confidence:.2f} for {action} {symbol}, converting to hold")
                action = "hold"

            valid_actions.append({
                "symbol": symbol,
                "action": action,
                "qty": qty,
                "price_estimate": price_estimate,
                "confidence": confidence,
                "reason": reason,
            })

            logger.info(f"  [{action.upper():4}] {symbol} x{qty} (conf={confidence:.2f}) — {reason}")

        except (TypeError, ValueError) as e:
            logger.warning(f"Skipping malformed action {item}: {e}")

    return valid_actions
