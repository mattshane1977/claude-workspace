"""
Places and manages orders via Alpaca Trading API.
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from loguru import logger
import config


def _client():
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)


def place_market_order(symbol: str, qty: int, side: str) -> dict | None:
    """
    Place a market order. side = 'buy' or 'sell'.
    Returns order dict or None on failure.
    """
    client = _client()
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )

    try:
        order = client.submit_order(request)
        logger.success(f"Order placed: {side.upper()} {qty} {symbol} | id={order.id}")
        return {
            "id": str(order.id),
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "status": str(order.status),
        }
    except Exception as e:
        logger.error(f"Order failed ({side} {qty} {symbol}): {e}")
        return None


def cancel_all_orders():
    """Cancel all open orders. Call on shutdown or circuit breaker trigger."""
    client = _client()
    try:
        client.cancel_orders()
        logger.info("All open orders cancelled")
    except Exception as e:
        logger.error(f"Failed to cancel orders: {e}")


def get_open_orders() -> list:
    """Return list of currently open orders."""
    client = _client()
    try:
        orders = client.get_orders()
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "qty": float(o.qty),
                "side": str(o.side),
                "status": str(o.status),
            }
            for o in orders
        ]
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return []


def is_market_open() -> bool:
    """Check if the US stock market is currently open."""
    client = _client()
    try:
        clock = client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        logger.error(f"Failed to check market clock: {e}")
        return False
