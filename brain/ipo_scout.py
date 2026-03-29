"""
IPO Scout — monitors SEC EDGAR for:
  1. New S-1 filings (companies going public), rated by LLM for investment interest
  2. Keyword scanner — finds companies mentioning SpaceX, NASA, defense, etc.
     in S-1 and 10-K/10-Q filings (surfaces hidden suppliers before media notices)

Runs daily at 7:00 AM ET so results are ready before market open.
"""
import json
import re
import time
from datetime import datetime, timedelta

import requests
from loguru import logger

import config
from brain import llm_stream

HEADERS = {"User-Agent": "trader-bot/1.0 research@localhost"}

# Keywords to scan for across all recent SEC filings
SPACE_DEFENSE_KEYWORDS = [
    "SpaceX", "Starlink", "NASA", "Rocket Lab", "Blue Origin",
    "satellite constellation", "launch vehicle", "reusable rocket",
    "hypersonic", "Department of Defense", "DoD contract",
    "defense contractor", "orbital", "spacecraft",
]

# SIC codes for space/defense industries (used to tag companies)
SPACE_DEFENSE_SICS = {
    "3760", "3761", "3764", "3769",  # missiles & space vehicles
    "3812",  # defense electronics / navigation
    "3489",  # ordnance
    "3812",  # search/detection/navigation systems
    "3699",  # electronic equipment (space hardware)
    "3812",  # guided missile systems
    "3728",  # aircraft parts (space-adjacent)
    "3812",  # aeronautical systems
}

IPO_SYSTEM_PROMPT = """You are an investment analyst specializing in pre-IPO companies and emerging space/defense technology plays.

You will receive a list of companies that have recently filed S-1 registration statements (going public) or are public companies with exposure to space, defense, or SpaceX/NASA supply chains.

Rate each for investment interest and hidden gem potential.

Your output MUST be valid JSON in exactly this format:
{
  "ratings": [
    {
      "id": <integer id from input>,
      "score": <float 0.0-1.0, higher = more interesting>,
      "verdict": "<Hot|Interesting|Pass>",
      "reason": "<2-3 sentence explanation — what they do, why interesting or not>"
    }
  ],
  "summary": "<one sentence overall observation>"
}

Include ALL provided companies. Weight heavily:
- Companies supplying SpaceX, Starlink, NASA, or DoD programs
- Pre-IPO companies (S-1 filers) in space, defense, AI, or deep tech
- Small/mid-cap public companies with undiscovered defense/space exposure
- Companies with government contracts (recurring revenue, defensible moat)
- Avoid rating highly: SPACs, blank-check companies, financial products, real estate

Output ONLY the JSON. No explanation, no markdown, no code blocks."""


# ── EDGAR helpers ─────────────────────────────────────────────────────────────

def _efts_search(forms: str, q: str = "", days: int = 7, size: int = 100) -> list[dict]:
    """Search EDGAR EFTS, return list of hit _source dicts with _id added."""
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    params = {
        "forms": forms,
        "dateRange": "custom",
        "startdt": start,
        "enddt": end,
    }
    if q:
        params["q"] = q
    try:
        r = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            logger.warning(f"EDGAR EFTS returned {r.status_code} for forms={forms} q={q!r}")
            return []
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        logger.info(f"EDGAR search forms={forms!r} q={q!r}: {total} total, returning {len(hits)}")
        result = []
        for h in hits:
            entry = dict(h.get("_source", {}))
            entry["_id"] = h.get("_id", "")
            result.append(entry)
        return result
    except Exception as e:
        logger.warning(f"EDGAR EFTS search failed: {e}")
        return []


def _get_filing_text(cik: str, accession: str, max_bytes: int = 80_000) -> str:
    """
    Fetch the first max_bytes of the main S-1 document and return extracted text.
    Tries to find the business/prospectus summary section.
    """
    try:
        acc_nd = accession.replace("-", "")
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nd}/{accession}-index.htm"
        r = requests.get(idx_url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return ""

        # Find the main document link (first .htm file in the index)
        links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"', r.text)
        if not links:
            return ""

        main_url = f"https://www.sec.gov{links[0]}"

        # Stream-fetch only the first max_bytes to avoid downloading 2MB+ files
        chunk = b""
        with requests.get(main_url, headers=HEADERS, timeout=15, stream=True) as resp:
            if resp.status_code != 200:
                return ""
            for data in resp.iter_content(chunk_size=8192):
                chunk += data
                if len(chunk) >= max_bytes:
                    break

        raw = chunk.decode("utf-8", errors="ignore")

        # Strip HTML tags and entities
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Try to find the prospectus summary / business overview section
        for marker in [
            "PROSPECTUS SUMMARY", "Prospectus Summary",
            "BUSINESS OVERVIEW", "Business Overview",
            "OUR BUSINESS", "Our Business",
            "ABOUT THE COMPANY", "About the Company",
        ]:
            idx = text.find(marker)
            if idx > 100:
                return text[idx: idx + 2500]

        # Fallback: skip boilerplate header, return meaningful text
        return text[600:3000]

    except Exception as e:
        logger.debug(f"Failed to fetch S-1 text for {cik}/{accession}: {e}")
        return ""


def _extract_cik(display_names: list) -> str:
    """Extract the first numeric CIK from EDGAR display_names list."""
    for name in display_names or []:
        match = re.search(r"CIK (\d+)", name)
        if match:
            return match.group(1).lstrip("0")
    return ""


def _extract_tickers(display_names: list) -> str:
    """Extract ticker symbols from display_names like 'Corp (TICK, TICKW) (CIK ...)'."""
    tickers = []
    for name in display_names or []:
        matches = re.findall(r"\(([A-Z]{1,5}(?:,\s*[A-Z]{1,5}W?)*)\)", name)
        for m in matches:
            if "CIK" not in m:
                tickers.extend([t.strip() for t in m.split(",")])
    return ", ".join(tickers[:3])


def _company_name(display_names: list) -> str:
    if not display_names:
        return "Unknown"
    name = display_names[0]
    # Strip CIK and ticker suffix
    name = re.sub(r"\s*\([A-Z,\s]+\)\s*\(CIK.*?\)", "", name)
    name = re.sub(r"\s*\(CIK.*?\)", "", name)
    return name.strip()


# ── Step 1: Recent S-1 filings ────────────────────────────────────────────────

def _get_recent_s1s(days: int = 30) -> list[dict]:
    """
    Fetch recent S-1 / S-1/A filings from EDGAR.
    Returns list of candidate dicts.
    """
    hits = _efts_search(forms="S-1,S-1/A", days=days)
    candidates = []
    seen_ciks = set()

    for h in hits:
        names = h.get("display_names", [])
        cik = _extract_cik(names)
        if not cik or cik in seen_ciks:
            continue
        seen_ciks.add(cik)

        accession = h.get("adsh", "")
        file_id = h.get("_id", "")
        edgar_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-','')}/{accession}-index.htm"
            if accession else ""
        )

        sics = h.get("sics", [])
        sic = sics[0] if sics else ""
        locs = h.get("biz_locations", [])
        location = next((l for l in locs if l), "")

        candidates.append({
            "company": _company_name(names),
            "ticker": _extract_tickers(names),
            "cik": cik,
            "filed_date": h.get("file_date", ""),
            "form_type": h.get("form", "S-1"),
            "sic": sic,
            "sic_description": "",  # enriched later if needed
            "location": location,
            "description": "",  # fetched selectively below
            "space_keywords": "",
            "edgar_url": edgar_url,
            "accession": accession,
        })

    logger.info(f"S-1 candidates: {len(candidates)} unique companies in last {days} days")
    return candidates


# ── Step 2: Space/defense keyword scanner ─────────────────────────────────────

def _get_keyword_candidates(days: int = 30) -> list[dict]:
    """
    Search EDGAR for recent filings (S-1, 10-K, 10-Q) mentioning
    space/defense keywords. Returns companies not already in S-1 list.
    """
    candidates = []
    seen_ciks = set()

    for keyword in SPACE_DEFENSE_KEYWORDS:
        time.sleep(0.3)  # respect EDGAR rate limit between keyword searches
        hits = _efts_search(
            forms="S-1,10-K,10-Q",
            q=f'"{keyword}"',
            days=days,
        )
        for h in hits:
            names = h.get("display_names", [])
            cik = _extract_cik(names)
            if not cik or cik in seen_ciks:
                continue
            seen_ciks.add(cik)

            accession = h.get("adsh", "")
            edgar_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-','')}/{accession}-index.htm"
                if accession else ""
            )
            sics = h.get("sics", [])
            locs = h.get("biz_locations", [])

            candidates.append({
                "company": _company_name(names),
                "ticker": _extract_tickers(names),
                "cik": cik,
                "filed_date": h.get("file_date", ""),
                "form_type": h.get("form", ""),
                "sic": sics[0] if sics else "",
                "sic_description": "",
                "location": next((l for l in locs if l), ""),
                "description": "",
                "space_keywords": keyword,
                "edgar_url": edgar_url,
                "accession": accession,
            })

    logger.info(f"Keyword scan candidates: {len(candidates)} unique companies")
    return candidates


# ── Step 3: Fetch business descriptions for high-interest candidates ───────────

def _enrich_descriptions(candidates: list[dict], max_fetch: int = 20) -> list[dict]:
    """
    Fetch the first portion of the main document for S-1 filers and
    keyword-matched companies. Limits total fetches to avoid overloading EDGAR.
    """
    fetched = 0
    for c in candidates:
        if fetched >= max_fetch:
            break
        if not c.get("accession") or not c.get("cik"):
            continue
        # Prioritize S-1 filers and space/defense keyword matches
        is_s1 = "S-1" in c.get("form_type", "")
        has_keyword = bool(c.get("space_keywords"))
        if not (is_s1 or has_keyword):
            continue

        text = _get_filing_text(c["cik"], c["accession"])
        if text:
            c["description"] = text[:1500]
            fetched += 1
        time.sleep(0.15)

    logger.info(f"Fetched descriptions for {fetched} candidates")
    return candidates


# ── Step 4: Deduplicate and merge ─────────────────────────────────────────────

def _merge_candidates(s1s: list[dict], keyword_hits: list[dict]) -> list[dict]:
    """Merge S-1 list and keyword hits, combining keywords for duplicates."""
    merged: dict[str, dict] = {}

    for c in s1s:
        key = c["cik"] or c["company"]
        merged[key] = c

    for c in keyword_hits:
        key = c["cik"] or c["company"]
        if key in merged:
            existing_kw = merged[key].get("space_keywords", "")
            new_kw = c.get("space_keywords", "")
            if new_kw and new_kw not in existing_kw:
                merged[key]["space_keywords"] = (
                    f"{existing_kw}, {new_kw}" if existing_kw else new_kw
                )
        else:
            merged[key] = c

    result = list(merged.values())
    # Sort: S-1 filers first, then by date descending
    result.sort(key=lambda x: (x.get("form_type", "") != "S-1", x.get("filed_date", "")), reverse=False)
    return result


# ── Step 5: LLM rating ────────────────────────────────────────────────────────

def _rate_with_llm(candidates: list[dict]) -> dict:
    """Send candidates to Ollama for IPO/space-play rating."""
    if not candidates:
        return {"ratings": [], "summary": "No candidates."}

    prompt_data = [
        {
            "id": i,
            "company": c["company"],
            "ticker": c["ticker"] or "Pre-IPO",
            "form_type": c["form_type"],
            "filed_date": c["filed_date"],
            "sector_sic": c["sic"],
            "location": c["location"],
            "space_defense_keywords": c["space_keywords"] or "none",
            "business_description": c["description"][:800] if c["description"] else "Not available",
        }
        for i, c in enumerate(candidates)
    ]

    prompt = (
        f"Rate these {len(prompt_data)} companies for investment interest "
        f"(pre-IPO plays and space/defense suppliers):\n\n"
        + json.dumps(prompt_data, indent=2)
    )

    logger.info(f"Sending {len(candidates)} IPO/space candidates to {config.OLLAMA_MODEL}...")
    raw = ""
    try:
        raw = llm_stream.chat(
            messages=[
                {"role": "system", "content": IPO_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 3000},
            label="IPO",
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
        logger.error(f"LLM returned invalid JSON for IPO scout: {e}")
        return {"ratings": [], "summary": "LLM parse error."}
    except Exception as e:
        logger.error(f"Ollama call failed for IPO scout: {e}")
        return {"ratings": [], "summary": str(e)}


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_ipo_scout(s1_days: int = 30, keyword_days: int = 30) -> list[dict]:
    """
    Full IPO/space-supplier scouting pipeline:
      1. Fetch recent S-1 filings (new IPO candidates)
      2. Keyword-scan EDGAR for space/defense supplier mentions
      3. Fetch business descriptions for priority candidates
      4. LLM rates all for investment interest
      5. Return sorted results (Hot first)
    """
    logger.info("=== IPO Scout starting ===")

    logger.info("Step 1: Fetching recent S-1 filings...")
    s1_candidates = _get_recent_s1s(days=s1_days)

    logger.info("Step 2: Scanning for space/defense keywords...")
    keyword_candidates = _get_keyword_candidates(days=keyword_days)

    logger.info("Step 3: Merging and enriching descriptions...")
    all_candidates = _merge_candidates(s1_candidates, keyword_candidates)

    if not all_candidates:
        logger.warning("IPO Scout: no candidates found")
        return []

    all_candidates = _enrich_descriptions(all_candidates, max_fetch=25)

    logger.info(f"Step 4: LLM rating {len(all_candidates)} candidates...")
    # Process in batches of 20 to stay within LLM context limits
    batch_size = 20
    all_ratings: dict[int, dict] = {}
    llm_summary = ""

    for batch_start in range(0, len(all_candidates), batch_size):
        batch = all_candidates[batch_start: batch_start + batch_size]
        # Re-index within batch
        batch_copy = [{**c} for c in batch]
        result = _rate_with_llm(batch_copy)
        for r in result.get("ratings", []):
            all_ratings[batch_start + r["id"]] = r
        if not llm_summary and result.get("summary"):
            llm_summary = result["summary"]
        if len(all_candidates) > batch_size:
            time.sleep(2)

    results = []
    for i, c in enumerate(all_candidates):
        rating = all_ratings.get(i, {})
        results.append({
            **c,
            "ai_score": rating.get("score"),
            "ai_verdict": rating.get("verdict", "Pass"),
            "ai_reason": rating.get("reason", ""),
            "llm_summary": llm_summary,
        })

    # Sort: Hot first, then by score
    verdict_order = {"Hot": 0, "Interesting": 1, "Pass": 2}
    results.sort(key=lambda x: (verdict_order.get(x.get("ai_verdict", "Pass"), 2), -(x.get("ai_score") or 0)))

    logger.info(f"=== IPO Scout complete: {len(results)} companies rated ===")
    return results
