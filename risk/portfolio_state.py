"""
Fetches current portfolio state from Alpaca: cash, positions, P&L.
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus
from loguru import logger
import config


def get_state() -> dict:
    """Return a snapshot of portfolio: cash, equity, positions, daily P&L."""
    client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    try:
        account = client.get_account()
        positions = client.get_all_positions()

        equity = float(account.equity)
        cash = float(account.cash)
        last_equity = float(account.last_equity)
        daily_pnl_pct = round((equity - last_equity) / last_equity * 100, 3) if last_equity else 0

        pos_list = []
        for p in positions:
            pos_list.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": round(float(p.unrealized_plpc) * 100, 3),
                "pct_of_portfolio": round(float(p.market_value) / equity * 100, 2) if equity else 0,
            })

        return {
            "equity": equity,
            "cash": cash,
            "cash_pct": round(cash / equity * 100, 2) if equity else 100,
            "daily_pnl_pct": daily_pnl_pct,
            "positions": pos_list,
            "num_positions": len(pos_list),
        }

    except Exception as e:
        logger.error(f"Failed to fetch portfolio state: {e}")
        return {}


def get_position(symbol: str) -> dict | None:
    """Return position for a specific symbol, or None if not held."""
    state = get_state()
    for p in state.get("positions", []):
        if p["symbol"] == symbol:
            return p
    return None


def count_trades_today() -> int:
    """Count orders placed today (filled or not)."""
    client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    try:
        from datetime import date
        today = date.today().isoformat()
        orders = client.get_orders(
            GetOrdersRequest(status=OrderStatus.ALL, after=today, limit=100)
        )
        return len(orders)
    except Exception as e:
        logger.error(f"Failed to count trades: {e}")
        return 0
