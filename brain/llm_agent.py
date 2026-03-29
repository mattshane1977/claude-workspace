"""
Sends market context to a local Ollama LLM and gets trade decisions.
"""
import json
from loguru import logger
import config
from brain import llm_stream

SYSTEM_PROMPT = """You are an autonomous stock trading agent. Your job is to analyze market data and make buy, sell, or hold decisions for a small retail portfolio.

You will receive:
- Portfolio state (cash, equity, current positions, daily P&L)
- Per-symbol market data: current price, technical indicators (RSI, MACD, EMA, Bollinger Bands, ATR), recent bar history
- Risk rules you must respect

Your output MUST be valid JSON in exactly this format:
{
  "actions": [
    {
      "symbol": "AAPL",
      "action": "buy" | "sell" | "hold",
      "qty": <integer, shares>,
      "price_estimate": <float, expected fill price>,
      "confidence": <float, 0.0-1.0>,
      "reason": "<brief reason>"
    }
  ],
  "summary": "<one sentence summary of your overall view>"
}

Trading rules you MUST follow:
- Only output symbols from the watchlist provided
- qty must be a positive integer
- If holding, include in actions with action="hold" and qty=0
- Do not recommend buying if cash is too low
- Do not recommend selling a symbol you have no position in
- Be conservative — protecting capital is more important than chasing gains
- RSI > 70 is overbought (consider sell/hold), RSI < 30 is oversold (consider buy)
- MACD histogram positive and rising = bullish momentum
- Price below BB lower band = oversold, above BB upper band = overbought
- When trend is "bearish" and RSI is falling, do not buy
- Factor in unrealized P&L on existing positions before selling

Output ONLY the JSON. No explanation, no markdown, no code blocks."""


def decide(context_text: str) -> dict:
    """
    Send context to Ollama and return parsed decision dict.
    Returns empty dict on failure.
    """
    prompt = f"Current market data and portfolio state:\n\n{context_text}\n\nMake your trading decisions now."

    logger.info(f"Sending context to {config.OLLAMA_MODEL} via Ollama...")

    raw = ""
    try:
        raw = llm_stream.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1, "num_predict": 1024},
            label="Trader",
        )
        raw = raw.strip()

        # Strip markdown code blocks if model wrapped output anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}\nRaw: {raw}")
        return {}
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return {}
