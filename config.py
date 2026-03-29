import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Alpaca
ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")

# Watchlist
WATCHLIST = os.getenv("WATCHLIST", "AAPL,MSFT,SPY").split(",")

# Decision loop
DECISION_INTERVAL_MINUTES = int(os.getenv("DECISION_INTERVAL_MINUTES", "15"))

# Risk guardrails
RISK = {
    "max_position_pct": 0.20,       # max 20% of portfolio in one stock
    "max_daily_loss_pct": 0.05,     # halt if down 5% today
    "max_drawdown_pct": 0.15,       # halt if down 15% from portfolio peak
    "max_trades_per_day": 10,       # circuit breaker on runaway loops
    "min_cash_reserve_pct": 0.10,   # always keep 10% cash
    "max_qty_per_order": 50,        # hard cap on shares per single order
}

# How many historical bars to feed the LLM
BARS_LOOKBACK = 30
BAR_TIMEFRAME = "15Min"            # 1Min, 5Min, 15Min, 1Hour, 1Day
