"""
Stock Scout — discovers "hidden gem" stocks using multiple signal sources:
  1. Yahoo Finance day-movers / trending stocks (discovery)
  2. SEC EDGAR Form 4 insider purchases (optional discovery — adds symbols not in Yahoo list)
  3. yfinance insider buying summary for all candidates (reliable enrichment signal)
  4. All candidates enriched with fundamentals + news, then rated by local Ollama LLM.
"""
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests
import yfinance as yf
from loguru import logger

import config
from brain import llm_stream

HEADERS = {"User-Agent": "trader-bot/1.0 research@localhost"}

SCOUT_SYSTEM_PROMPT = """You are a stock research analyst specializing in finding "hidden gem" stocks — companies with strong fundamentals or momentum that haven't yet attracted mainstream attention.

You will receive a list of stock candidates with their financial data, recent news, and insider-buying signals. Analyze each one and identify which have the most upside potential.

Your output MUST be valid JSON in exactly this format:
{
  "gems": [
    {
      "symbol": "XYZ",
      "score": <float 0.0-1.0, higher = more promising>,
      "verdict": "<Bullish|Neutral|Bearish>",
      "reason": "<2-3 sentence explanation of potential>"
    }
  ],
  "summary": "<one sentence overall observation about this batch of stocks>"
}

Include ALL provided symbols in your output. When rating, weight these signals heavily:
- INSIDER BUYING: Officers/directors purchasing their own stock with personal money is one of the strongest bullish signals. Score these higher.
- Stocks trading well below 52-week high but showing reversal signs
- Reasonable P/E vs sector peers, or high growth justifying premium
- Strong recent news catalysts (earnings beats, contracts, new products)
- Volume significantly above average — signals institutional accumulation
- Small/mid-cap stocks under-covered by mainstream analysts

Output ONLY the JSON. No explanation, no markdown, no code blocks."""


# ── Signal Source 1: Yahoo Finance movers ────────────────────────────────────

def _get_trending_symbols(count: int = 25) -> list[str]:
    """Fetch trending/gaining US equity symbols from Yahoo Finance."""
    try:
        r = requests.get(
            "https://query2.finance.yahoo.com/v8/finance/trending/US",
            params={"count": count, "lang": "en-US"},
            headers=HEADERS,
            timeout=10,
        )
        data = r.json()
        symbols = [q.get("symbol", "") for q in data["finance"]["result"][0]["quotes"]]
        symbols = [s for s in symbols if s and s.replace("-", "").isalpha() and len(s) <= 5]
        if symbols:
            logger.info(f"Trending symbols ({len(symbols)}): {symbols}")
            return symbols
    except Exception as e:
        logger.debug(f"Trending endpoint failed: {e}")

    # Fallback: day gainers screener
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
            params={"scrIds": "day_gainers", "count": count, "lang": "en-US", "region": "US"},
            headers=HEADERS,
            timeout=10,
        )
        data = r.json()
        symbols = [q.get("symbol", "") for q in data["finance"]["result"][0]["quotes"] if q.get("symbol", "").isalpha()]
        logger.info(f"Day-gainers symbols ({len(symbols)}): {symbols}")
        return symbols
    except Exception as e:
        logger.error(f"Yahoo Finance symbol fetch failed: {e}")
        return []


# ── Signal Source 2: SEC EDGAR Form 4 insider purchases (discovery only) ─────

def _get_edgar_insider_symbols(days: int = 7, max_filings: int = 80) -> dict[str, list[dict]]:
    """
    Fetch recent insider purchases from SEC EDGAR Form 4 filings.
    Used for DISCOVERY — adds symbols not already in the Yahoo list.
    Returns {symbol: [{"insider", "title", "shares", "value", "date"}]}
    Falls back to empty dict on any error.
    """
    signals: dict[str, list[dict]] = {}
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    try:
        r = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"forms": "4", "dateRange": "custom", "startdt": start, "enddt": end},
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            logger.warning(f"EDGAR EFTS returned {r.status_code}, skipping insider discovery")
            return signals

        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        logger.info(f"EDGAR: {total} Form 4 filings in last {days} days, scanning {min(len(hits), max_filings)}")
    except Exception as e:
        logger.warning(f"EDGAR Form 4 search failed: {e} — insider discovery skipped")
        return signals

    for hit in hits[:max_filings]:
        try:
            src = hit.get("_source", {})
            adsh = src.get("adsh", "")
            ciks = src.get("ciks", [])
            file_id = hit.get("_id", "")
            file_date = src.get("file_date", "")

            if not adsh or not ciks or ":" not in file_id:
                continue
            xml_filename = file_id.split(":", 1)[1]
            if not xml_filename.endswith(".xml"):
                continue

            cik = ciks[0].lstrip("0")
            xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-','')}/{xml_filename}"

            rx = requests.get(xml_url, headers=HEADERS, timeout=8)
            if rx.status_code != 200:
                continue

            root = ET.fromstring(rx.content)
            ticker = root.findtext(".//issuerTradingSymbol", "").strip().upper()
            if not ticker or not ticker.isalpha() or len(ticker) > 5:
                continue

            is_officer = root.findtext(".//isOfficer", "0").strip() == "1"
            is_director = root.findtext(".//isDirector", "0").strip() == "1"
            if not (is_officer or is_director):
                continue

            insider_name = root.findtext(".//rptOwnerName", "").strip()
            officer_title = root.findtext(".//officerTitle", "").strip()

            for txn in root.findall(".//nonDerivativeTransaction"):
                if txn.findtext(".//transactionCode", "").strip() != "P":
                    continue
                try:
                    shares = int(float(txn.findtext(".//transactionShares/value", "0")))
                except Exception:
                    shares = 0
                if shares <= 0:
                    continue

                price_raw = txn.findtext(".//transactionPricePerShare/value", "").strip()
                value_str = ""
                if price_raw:
                    try:
                        value_str = f"${int(float(price_raw) * shares):,}"
                    except Exception:
                        pass

                role = officer_title or ("Director" if is_director else "Officer")
                if ticker not in signals:
                    signals[ticker] = []
                signals[ticker].append({
                    "insider": insider_name,
                    "title": role,
                    "shares": shares,
                    "value": value_str,
                    "date": file_date,
                    "source": "EDGAR",
                })
                logger.debug(f"EDGAR insider buy: {insider_name} ({role}) bought {shares:,} shares of {ticker}")

            time.sleep(0.12)  # respect SEC rate limit (max 10 req/sec)

        except Exception as e:
            logger.debug(f"Failed to parse Form 4 {hit.get('_id', '')}: {e}")
            continue

    logger.info(f"EDGAR insider signals: {len(signals)} symbols with officer/director purchases")
    return signals


# ── Enrich candidates: fundamentals + news + yfinance insider signal ──────────

def _parse_news(ticker_obj) -> str:
    headlines = []
    try:
        for n in (ticker_obj.news or [])[:3]:
            title = ""
            if isinstance(n.get("content"), dict):
                title = n["content"].get("title", "")
            if not title:
                title = n.get("title", "")
            if title:
                headlines.append(title)
    except Exception:
        pass
    return " | ".join(headlines)[:300]


def _get_yf_insider_signal(ticker_obj) -> list[str]:
    """
    Check yfinance insider_purchases summary for net buying over last 6 months.
    Returns a list of signal strings (empty if no notable buying).
    """
    try:
        ip = ticker_obj.insider_purchases
        if ip is None or ip.empty:
            return []

        rows = ip.set_index(ip.columns[0])
        purchases = float(rows.loc["Purchases", "Shares"]) if "Purchases" in rows.index else 0
        sales = float(rows.loc["Sales", "Shares"]) if "Sales" in rows.index else 0

        if purchases > 0 and purchases > sales:
            net = int(purchases - sales)
            return [f"Net insider buying (6mo): {int(purchases):,} purchased vs {int(sales):,} sold — net +{net:,} shares"]
    except Exception:
        pass
    return []


def enrich_candidates(symbols: list[str], watchlist: list[str], edgar_signals: dict = None) -> list[dict]:
    """
    Fetch fundamentals + news + insider signals for each symbol.
    Filters out watchlist stocks, pennies (<$2), and micro-caps (<$50M).
    edgar_signals: dict of {symbol: [trade_info]} from EDGAR (optional)
    """
    watchlist_upper = {s.upper() for s in watchlist}
    edgar_signals = edgar_signals or {}
    candidates = []

    for symbol in symbols:
        if symbol.upper() in watchlist_upper:
            continue
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            market_cap = info.get("marketCap") or 0

            if not price or price < 2.0:
                continue
            if market_cap and market_cap < 50_000_000:
                continue

            avg_vol = info.get("averageVolume") or 0
            cur_vol = info.get("volume") or 0
            vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol else None

            # Build signals list: EDGAR first, then yfinance
            signals_list = []
            for trade in edgar_signals.get(symbol.upper(), []):
                signals_list.append(
                    f"{trade['insider']} ({trade['title']}) bought {trade['shares']:,} shares"
                    + (f" {trade['value']}" if trade.get("value") else "")
                    + f" [{trade['date']}] (SEC Form 4)"
                )
            signals_list.extend(_get_yf_insider_signal(ticker))

            candidates.append({
                "symbol": symbol,
                "company": info.get("shortName", symbol),
                "sector": info.get("sector", "Unknown"),
                "price": round(price, 2),
                "market_cap": market_cap,
                "pe_ratio": info.get("trailingPE"),
                "week52_high": info.get("fiftyTwoWeekHigh"),
                "week52_low": info.get("fiftyTwoWeekLow"),
                "volume_vs_avg": vol_ratio,
                "news_snippet": _parse_news(ticker),
                "signals": json.dumps(signals_list) if signals_list else None,
            })
            if signals_list:
                logger.info(f"Insider signal on {symbol}: {signals_list[0]}")
            else:
                logger.debug(f"Enriched {symbol}: {info.get('shortName')}")
        except Exception as e:
            logger.warning(f"Skipping {symbol}: {e}")
            continue

    logger.info(f"{len(candidates)} candidates after filtering (from {len(symbols)} raw)")
    return candidates


# ── LLM rating ────────────────────────────────────────────────────────────────

def rate_with_llm(candidates: list[dict]) -> dict:
    """Send enriched candidates to Ollama for gem scoring."""
    if not candidates:
        return {"gems": [], "summary": "No candidates to analyze."}

    prompt_data = []
    for c in candidates:
        entry = {
            "symbol": c["symbol"],
            "company": c["company"],
            "sector": c["sector"],
            "price": c["price"],
            "market_cap_M": round(c["market_cap"] / 1_000_000) if c["market_cap"] else None,
            "pe_ratio": c["pe_ratio"],
            "week52_range": f"{c['week52_low']} – {c['week52_high']}",
            "volume_vs_avg": c["volume_vs_avg"],
            "recent_news": c["news_snippet"],
        }
        if c.get("signals"):
            try:
                entry["insider_signals"] = json.loads(c["signals"])
            except Exception:
                pass
        prompt_data.append(entry)

    prompt = (
        f"Analyze these {len(prompt_data)} stocks and rate their hidden-gem potential:\n\n"
        + json.dumps(prompt_data, indent=2)
    )

    logger.info(f"Sending {len(candidates)} candidates to {config.OLLAMA_MODEL} for gem rating...")
    raw = ""
    try:
        raw = llm_stream.chat(
            messages=[
                {"role": "system", "content": SCOUT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 2048},
            label="Scout",
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON for scout: {e}")
        return {"gems": [], "summary": "LLM parse error."}
    except Exception as e:
        logger.error(f"Ollama call failed for scout: {e}")
        return {"gems": [], "summary": str(e)}


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_scout(watchlist: list[str]) -> list[dict]:
    """
    Full scouting pipeline:
      1. Fetch Yahoo Finance trending/movers
      2. Fetch SEC EDGAR Form 4 insider purchases (optional, adds extra symbols)
      3. Merge symbol lists and enrich with fundamentals + news + yfinance insider check
      4. LLM rates each for hidden-gem potential
      5. Return sorted results (best gems first)
    """
    logger.info("=== Stock Scout starting ===")

    logger.info("Step 1: Fetching Yahoo Finance trending stocks...")
    yahoo_symbols = _get_trending_symbols(count=25)

    logger.info("Step 2: Fetching SEC EDGAR Form 4 insider purchases (discovery)...")
    edgar_signals = _get_edgar_insider_symbols(days=7, max_filings=80)

    # Merge: Yahoo movers + EDGAR insider-bought symbols (not on watchlist)
    watchlist_upper = {s.upper() for s in watchlist}
    edgar_only = [s for s in edgar_signals if s not in watchlist_upper and s not in yahoo_symbols]
    all_symbols = list(dict.fromkeys(yahoo_symbols + edgar_only))

    if edgar_signals:
        logger.info(f"EDGAR added {len(edgar_only)} new symbols not in Yahoo list")

    logger.info(f"Step 3: Enriching {len(all_symbols)} total candidates...")
    candidates = enrich_candidates(all_symbols, watchlist, edgar_signals)

    if not candidates:
        logger.warning("Scout: no candidates survived filtering")
        return []

    logger.info(f"Step 4: LLM rating {len(candidates)} candidates...")
    ratings = rate_with_llm(candidates)
    gem_map = {g["symbol"]: g for g in ratings.get("gems", [])}
    llm_summary = ratings.get("summary", "")

    results = []
    for c in candidates:
        gem = gem_map.get(c["symbol"], {})
        results.append({
            **c,
            "ai_score": gem.get("score"),
            "ai_verdict": gem.get("verdict", "Neutral"),
            "ai_reason": gem.get("reason", ""),
            "llm_summary": llm_summary,
        })

    results.sort(key=lambda x: x.get("ai_score") or 0, reverse=True)
    logger.info(f"=== Scout complete: {len(results)} stocks rated ===")
    return results
