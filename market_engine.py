"""Kalshi weather market discovery, parsing, and fee-aware value calculations."""
from __future__ import annotations
import re
import requests
from datetime import datetime

BASE = "https://external-api.kalshi.com/trade-api/v2"

# Airport/station coordinates are starting points only. Verify the settlement source
# and station in every Kalshi market's rules before relying on a signal.
CITIES = {
    "NYC": ("New York", 40.7812, -73.9665, "America/New_York"),
    "NY": ("New York", 40.7812, -73.9665, "America/New_York"),
    "CHI": ("Chicago", 41.9950, -87.9336, "America/Chicago"),
    "CHICAGO": ("Chicago", 41.9950, -87.9336, "America/Chicago"),
    "LAX": ("Los Angeles", 33.9425, -118.4081, "America/Los_Angeles"),
    "LA": ("Los Angeles", 33.9425, -118.4081, "America/Los_Angeles"),
    "MIA": ("Miami", 25.7959, -80.2870, "America/New_York"),
    "PHIL": ("Philadelphia", 39.8721, -75.2411, "America/New_York"),
    "PHX": ("Phoenix", 33.4342, -112.0116, "America/Phoenix"),
    "AUS": ("Austin", 30.1975, -97.6664, "America/Chicago"),
    "HOU": ("Houston", 29.9902, -95.3368, "America/Chicago"),
    "DEN": ("Denver", 39.8561, -104.6737, "America/Denver"),
    "BOS": ("Boston", 42.3656, -71.0096, "America/New_York"),
    "ATL": ("Atlanta", 33.6407, -84.4277, "America/New_York"),
    "SEA": ("Seattle", 47.4502, -122.3088, "America/Los_Angeles"),
    "SFO": ("San Francisco", 37.6213, -122.3790, "America/Los_Angeles"),
    "DFW": ("Dallas", 32.8998, -97.0403, "America/Chicago"),
    "LAS": ("Las Vegas", 36.0840, -115.1537, "America/Los_Angeles"),
    "MSP": ("Minneapolis", 44.8848, -93.2223, "America/Chicago"),
    "DCA": ("Washington DC", 38.8512, -77.0402, "America/New_York"),
    "RDU": ("Raleigh", 35.8776, -78.7875, "America/New_York"),
}

WEATHER_WORDS = ("temperature", "highest temp", "lowest temp", "rain", "rainfall", "precipitation", "snow", "snowfall", "weather")


def get_json(url, params=None, timeout=18):
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "KalshiWeatherEdge/0.2 personal research"})
    r.raise_for_status()
    return r.json()


def fetch_open_weather_markets(max_pages=20):
    """Fetch open market pages and retain likely weather markets across series."""
    found, cursor, seen = [], None, set()
    for _ in range(max_pages):
        params = {"status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        payload = get_json(f"{BASE}/markets", params)
        for m in payload.get("markets", []):
            ticker = str(m.get("ticker", "")).upper()
            text = " ".join(str(m.get(k, "")) for k in ("title", "subtitle", "yes_sub_title")).lower()
            # Include known weather series plus weather-labelled titles. This avoids
            # assuming every future weather series shares today's ticker prefix.
            known = ticker.startswith(("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP", "KXPRECIP"))
            if (known or any(w in text for w in WEATHER_WORDS)) and ticker not in seen:
                m["_weather_kind"] = classify_market(m)
                found.append(m)
                seen.add(ticker)
        cursor = payload.get("cursor")
        if not cursor:
            break
    return found


def classify_market(m):
    t = (str(m.get("ticker", "")) + " " + str(m.get("title", "")) + " " + str(m.get("subtitle", ""))).lower()
    if any(x in t for x in ("snow", "snowfall")): return "snow"
    if any(x in t for x in ("rain", "rainfall", "precipitation")): return "precipitation"
    if "lowest" in t or "low temperature" in t or "kxlow" in t: return "low_temperature"
    if "highest" in t or "high temperature" in t or "kxhigh" in t: return "high_temperature"
    if "temperature" in t or "kxtemp" in t: return "temperature"
    return "other_weather"


def map_city(m):
    hay = " ".join(str(m.get(k, "")) for k in ("ticker", "title", "subtitle", "yes_sub_title")).upper()
    ticker = str(m.get("ticker", "")).upper()
    series_match = re.search(r"KX(?:HIGH|LOW|RAIN|SNOW|TEMP|PRECIP)([A-Z]+)", ticker)
    if series_match:
        series_code = series_match.group(1)
        if series_code in CITIES:
            return CITIES[series_code]
        for code in sorted(CITIES, key=len, reverse=True):
            if series_code.endswith(code):
                return CITIES[code]
    for code in sorted(CITIES, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(code)}(?![A-Z])", hay) or ticker.endswith(code):
            return CITIES[code]
    return None


def market_target_date(m):
    ticker = str(m.get("ticker", "")).upper()
    match = re.search(r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})-", ticker)
    if match:
        try:
            return datetime.strptime("20" + match.group(1) + match.group(2) + match.group(3), "%Y%b%d").date().isoformat()
        except ValueError:
            pass
    for field in ("close_time", "expiration_time"):
        value = m.get(field)
        if value:
            try: return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
            except Exception: pass
    return None


def market_bracket(m):
    """Parse common numeric temperature / precipitation contract thresholds."""
    # Parse the YES outcome label first; titles often contain dates that look numeric.
    fields = [m.get("yes_sub_title"), m.get("subtitle"), m.get("title")]
    for raw in fields:
        if not raw:
            continue
        text = str(raw)
        low = text.lower().replace("−", "-")
        nums = [float(x) for x in re.findall(r"(?<!\d)(-?\d{1,3}(?:\.\d+)?)\s*°?", text)]
        unit = "F" if "°f" in low or "fahrenheit" in low else ("C" if "°c" in low or "celsius" in low else None)
        if len(nums) >= 2:
            return {"kind": "range", "lo": min(nums[:2]), "hi": max(nums[:2]), "unit": unit}
        if len(nums) == 1:
            n = nums[0]
            if any(x in low for x in ("or above", "or higher", "at least", "or more", "greater than")):
                return {"kind": "above", "lo": n, "hi": None, "unit": unit}
            if any(x in low for x in ("or below", "or lower", "at most", "or less", "less than")):
                return {"kind": "below", "lo": None, "hi": n, "unit": unit}
    return None


def dollars(m, field):
    v = m.get(field)
    if v is None: return None
    try: return float(v) if "_dollars" in field else float(v) / 100
    except (TypeError, ValueError): return None


def best_side_prices(m):
    # The public markets response may expose dollar or integer-cent fields.
    ya = dollars(m, "yes_ask_dollars")
    yb = dollars(m, "yes_bid_dollars")
    na = dollars(m, "no_ask_dollars")
    nb = dollars(m, "no_bid_dollars")
    if ya is None and m.get("yes_ask") is not None: ya = float(m["yes_ask"])/100
    if yb is None and m.get("yes_bid") is not None: yb = float(m["yes_bid"])/100
    if na is None and m.get("no_ask") is not None: na = float(m["no_ask"])/100
    if nb is None and m.get("no_bid") is not None: nb = float(m["no_bid"])/100
    # Complementary best bid can approximate the opposite ask, but flag it as derived.
    if na is None and yb is not None: na = 1-yb
    if ya is None and nb is not None: ya = 1-nb
    return ya, yb, na, nb


def fee_per_contract(price, fee_rate=0.07):
    """Approximate taker fee: rate * p * (1-p); user-adjustable, not official quote."""
    if price is None: return None
    return max(0.0, float(fee_rate) * price * (1-price))


def trade_metrics(prob_yes, yes_ask, no_ask, fee_rate=0.07):
    out = {}
    for side, p_model, cost in (("YES", prob_yes, yes_ask), ("NO", 1-prob_yes if prob_yes is not None else None, no_ask)):
        if p_model is None or cost is None or cost <= 0 or cost >= 1: continue
        fee = fee_per_contract(cost, fee_rate)
        ev = p_model - cost - fee
        out[side] = {"model_probability": p_model, "entry_price": cost, "fee_estimate": fee,
                     "ev_per_contract": ev, "roi_on_cost": ev/cost if cost else None}
    return out
