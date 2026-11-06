"""Kalshi weather market discovery, parsing, and fee-aware value calculations.

READ-ONLY: this module only issues public GET requests. It never authenticates,
never touches an account, and never places orders.

Discovery strategy (v0.3)
-------------------------
The old code scanned the general open-market feed. That feed also contains a very
large number of multivariate ("combo"/parlay) markets, so weather contracts were
never reached within the page cap. Discovery now works by *series*:

  1. GET /series?category=Climate and Weather   -> list weather series tickers
  2. GET /markets?series_ticker=<T>&status=open -> open markets for each series
     (paginated; falls back to /events?...&with_nested_markets=true on error)
  3. If step 1 fails or is empty -> probe candidate series tickers built from the
     city table (clearly labelled "probe" in diagnostics).
  4. If everything above yields zero markets -> scan the general feed with
     mve_filter=exclude and a strict weather filter.

Every step is recorded in a diagnostics dict so an API failure can never be
mistaken for "there are no weather markets".
"""
from __future__ import annotations

import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

# Both hosts are documented by Kalshi as production servers for the same API.
BASES = (
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://external-api.kalshi.com/trade-api/v2",
)
BASE = BASES[0]  # kept for backward compatibility
WEATHER_CATEGORY = "Climate and Weather"
SEED_SERIES = ("KXHIGHNY",)  # documented in Kalshi's quick-start; always queried

CONNECT_TIMEOUT = 6
READ_TIMEOUT = 20
USER_AGENT = "KalshiWeatherEdge/0.3 personal read-only research"

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
# Full city names used only as a text fallback (never short codes like LA / NY).
CITY_NAMES = {
    "NEW YORK": "NYC", "CHICAGO": "CHI", "LOS ANGELES": "LAX", "MIAMI": "MIA",
    "PHILADELPHIA": "PHIL", "PHOENIX": "PHX", "AUSTIN": "AUS", "HOUSTON": "HOU",
    "DENVER": "DEN", "BOSTON": "BOS", "ATLANTA": "ATL", "SEATTLE": "SEA",
    "SAN FRANCISCO": "SFO", "DALLAS": "DFW", "LAS VEGAS": "LAS",
    "MINNEAPOLIS": "MSP", "WASHINGTON": "DCA", "RALEIGH": "RDU",
}

KNOWN_PREFIXES = ("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP", "KXPRECIP")
TEMP_KINDS = ("high_temperature", "low_temperature", "temperature")
MODELED_KINDS = TEMP_KINDS + ("precipitation", "snow")
WEATHER_RE = re.compile(
    r"\b(temperature|rainfall|snowfall|precipitation|rain|snow|weather|hurricane)\b", re.I)
# Kept for backward compatibility with older imports.
WEATHER_WORDS = ("temperature", "highest temp", "lowest temp", "rain", "rainfall",
                 "precipitation", "snow", "snowfall", "weather")


# --------------------------------------------------------------------------- HTTP
class ApiError(Exception):
    def __init__(self, message, status=None, url=None, failover=False):
        super().__init__(message)
        self.status, self.url, self.failover = status, url, failover


class MarketDiscoveryError(RuntimeError):
    """Raised by fetch_open_weather_markets() when discovery failed outright."""


def get_json(url, params=None, timeout=18):
    """Simple one-shot GET (kept for backward compatibility)."""
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()


class KalshiClient:
    """Small public-API client: timeouts, throttling, 429/5xx backoff, host failover."""

    def __init__(self, bases=BASES, max_rps=8.0, max_retries=4, session=None):
        self.bases = tuple(bases)
        self.session = session or requests.Session()
        try:
            self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        except Exception:
            pass
        self.max_retries = max_retries
        self.min_interval = 1.0 / max_rps
        self._lock = threading.Lock()
        self._next = 0.0
        self.base_index = 0
        self.stats = {"requests": 0, "retries": 0, "rate_limited": 0}

    @property
    def base_in_use(self):
        return self.bases[self.base_index]

    def _wait(self):
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.min_interval
        if slot > now:
            time.sleep(slot - now)

    def _get_one(self, base, path, params):
        url = f"{base}{path}"
        attempt = 0
        while True:
            self._wait()
            self.stats["requests"] += 1
            try:
                r = self.session.get(url, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.max_retries:
                    attempt += 1
                    self.stats["retries"] += 1
                    time.sleep(min(2 ** attempt, 8) + random.random() * 0.3)
                    continue
                raise ApiError(f"network error ({type(e).__name__}): {e}", None, url, failover=True)
            code = r.status_code
            if code == 429 or code >= 500:
                if code == 429:
                    self.stats["rate_limited"] += 1
                if attempt < self.max_retries:
                    attempt += 1
                    self.stats["retries"] += 1
                    try:
                        delay = float(r.headers.get("Retry-After", ""))
                    except (TypeError, ValueError):
                        delay = min(2 ** attempt, 16)
                    time.sleep(min(delay, 30) + random.random() * 0.3)
                    continue
                raise ApiError(f"HTTP {code} after {attempt} retries", code, url, failover=True)
            if code >= 400:
                snippet = (r.text or "")[:160].replace("\n", " ")
                raise ApiError(f"HTTP {code}: {snippet}", code, url, failover=(code == 403))
            try:
                return r.json()
            except ValueError:
                raise ApiError("response was not valid JSON (possible block page)", code, url, failover=True)

    def get(self, path, params=None):
        order = list(range(len(self.bases)))
        order = order[self.base_index:] + order[:self.base_index]
        last = None
        for bi in order:
            try:
                data = self._get_one(self.bases[bi], path, params)
                self.base_index = bi
                return data
            except ApiError as e:
                last = e
                if not e.failover:
                    raise
        raise last

    def paginate(self, path, params, items_key, max_pages):
        """Follow cursors. Returns (items, pages, truncated, error)."""
        items, pages, cursor, seen, error = [], 0, None, set(), None
        while pages < max_pages:
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            try:
                data = self.get(path, p)
            except ApiError as e:
                error = e
                break
            pages += 1
            items.extend(data.get(items_key) or [])
            cursor = data.get("cursor") or None
            if not cursor or cursor in seen:
                cursor = None
                break
            seen.add(cursor)
        return items, pages, bool(cursor) and pages >= max_pages, error


# ----------------------------------------------------------------- classification
def _series_of(m):
    return str(m.get("ticker", "")).upper().split("-")[0]


def _text(m, keys=("ticker", "title", "subtitle", "yes_sub_title", "_series_title")):
    return " ".join(str(m.get(k, "") or "") for k in keys).lower()


def is_weather_candidate(m):
    """Strict filter used only when scanning the general feed (fallback path)."""
    series = _series_of(m)
    if series.startswith(KNOWN_PREFIXES):
        return True
    return bool(WEATHER_RE.search(" ".join(str(m.get(k, "") or "") for k in
                                           ("title", "subtitle", "yes_sub_title"))))


def classify_market(m):
    series = _series_of(m)
    t = _text(m)
    if "SNOW" in series or re.search(r"\bsnow(fall)?\b", t): return "snow"
    if series.startswith(("KXRAIN", "KXPRECIP")) or re.search(r"\b(rain|rainfall|precipitation)\b", t):
        return "precipitation"
    if series.startswith("KXLOW") or "lowest" in t or "low temperature" in t: return "low_temperature"
    if series.startswith("KXHIGH") or "highest" in t or "high temperature" in t: return "high_temperature"
    if "temperature" in t or series.startswith("KXTEMP"): return "temperature"
    return "other_weather"


def map_city(m):
    """Map a market to a city using the *series code*, never loose suffix matching.

    (The old suffix match turned New Orleans ...NOLA into Los Angeles "LA".)
    """
    series = _series_of(m)
    sm = re.match(r"KX(?:HIGH|LOW|RAIN|SNOW|TEMP|PRECIP)([A-Z]+)$", series)
    if sm:
        code = sm.group(1)
        for cand in (code, code[1:] if code.startswith("T") else None):
            if cand and cand in CITIES:
                return CITIES[cand]
    hay = " ".join(str(m.get(k, "") or "") for k in
                   ("title", "subtitle", "yes_sub_title", "_series_title", "_event_title")).upper()
    for name, code in CITY_NAMES.items():
        if name in hay:
            return CITIES[code]
    for code in sorted((c for c in CITIES if len(c) >= 3), key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(code)}(?![A-Z])", hay):
            return CITIES[code]
    return None


_MON = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"


def market_target_date(m):
    """Target date for DAILY contracts. Monthly/weekly contracts return None.

    Daily tickers look like KXHIGHNY-26SEP14-T81 (YY MON DD). Monthly ones have no
    day (e.g. ...-26SEP-...) and must not be modelled as a single day.
    """
    ticker = str(m.get("ticker", "")).upper()
    freq = str(m.get("_frequency") or "").lower()
    if freq in ("weekly", "monthly", "annual", "yearly", "quarterly"):
        return None
    match = re.search(rf"-(\d{{2}})({_MON})(\d{{2}})(?:-|$)", ticker)
    if match:
        try:
            return datetime.strptime("20" + match.group(1) + match.group(2) + match.group(3), "%Y%b%d").date().isoformat()
        except ValueError:
            pass
    if re.search(rf"-(\d{{2}})({_MON})(?:-|$)", ticker):
        return None  # month-only ticker
    for field in ("close_time", "expiration_time"):
        value = m.get(field)
        if value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------- brackets
_ABOVE = re.compile(r"\b(or above|or higher|or more|at least|greater than|more than|higher than|above|over)\b")
_BELOW = re.compile(r"\b(or below|or lower|or less|at most|less than|fewer than|lower than|below|under)\b")


def _f(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _unit(m):
    blob = " ".join(str(m.get(k, "") or "") for k in ("yes_sub_title", "subtitle", "title", "rules_primary")).lower()
    if "°f" in blob or "fahrenheit" in blob: return "F"
    if "°c" in blob or "celsius" in blob: return "C"
    return None


def _bracket_from_text(m):
    fields = [m.get("yes_sub_title"), m.get("subtitle")]
    if not m.get("_title_synthetic"):
        fields.append(m.get("title"))
    unit = _unit(m)
    for raw in fields:
        if not raw:
            continue
        text = str(raw)
        low = text.lower().replace("−", "-")
        nums = [float(x) for x in re.findall(r"(?<!\d)(-?\d{1,3}(?:\.\d+)?)\s*°?", low)]
        if len(nums) >= 2:
            return {"kind": "range", "lo": min(nums[:2]), "hi": max(nums[:2]), "unit": unit, "source": "label"}
        if len(nums) == 1:
            n = nums[0]
            if _ABOVE.search(low):
                return {"kind": "above", "lo": n, "hi": None, "unit": unit, "source": "label"}
            if _BELOW.search(low):
                return {"kind": "below", "lo": None, "hi": n, "unit": unit, "source": "label"}
    return None


def _bracket_from_strikes(m):
    """Use Kalshi's structured strike fields (strike_type / floor_strike / cap_strike).

    Temperature contracts settle on whole degrees, so "greater than 81" means >= 82
    and "less than 70" means <= 69. Continuous measures (rain/snow) are left as-is.
    """
    st, fl, cp = m.get("strike_type"), _f(m.get("floor_strike")), _f(m.get("cap_strike"))
    if not st:
        return None
    step = 1.0 if classify_market(m) in TEMP_KINDS else 0.0
    unit = _unit(m)
    if st == "between" and fl is not None and cp is not None:
        return {"kind": "range", "lo": fl, "hi": cp, "unit": unit, "source": "strike"}
    if st == "greater" and fl is not None:
        return {"kind": "above", "lo": fl + step, "hi": None, "unit": unit, "source": "strike"}
    if st == "greater_or_equal" and fl is not None:
        return {"kind": "above", "lo": fl, "hi": None, "unit": unit, "source": "strike"}
    if st == "less" and cp is not None:
        return {"kind": "below", "lo": None, "hi": cp - step, "unit": unit, "source": "strike"}
    if st == "less_or_equal" and cp is not None:
        return {"kind": "below", "lo": None, "hi": cp, "unit": unit, "source": "strike"}
    return None


def market_bracket(m):
    """Parse the YES-outcome threshold. Label text first, structured strikes as fallback."""
    return _bracket_from_text(m) or _bracket_from_strikes(m)


def _bracket_conflict(a, b):
    return bool(a and b and (a["kind"] != b["kind"] or a.get("lo") != b.get("lo") or a.get("hi") != b.get("hi")))


# ------------------------------------------------------------------------ prices
def dollars(m, field):
    v = m.get(field)
    if v is None: return None
    try: return float(v) if "_dollars" in field else float(v) / 100
    except (TypeError, ValueError): return None


def best_side_prices(m):
    # Current Kalshi responses expose *_dollars strings; integer-cent fields are legacy.
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


# ------------------------------------------------------------------ normalisation
def normalize_market(m, series=None):
    """Fill gaps left by Kalshi's deprecated fields and add parsed helper fields.

    Kalshi marks `title` / `subtitle` as deprecated (they may be absent), and
    volume now arrives as the string `volume_fp`. Existing app code that reads
    m["title"] or m["volume"] keeps working because we populate them.
    Added keys all start with "_".
    """
    series = series or {}
    ticker = str(m.get("ticker", ""))
    ysub = str(m.get("yes_sub_title") or "")
    m["_series_ticker"] = series.get("ticker") or _series_of(m)
    m["_series_title"] = series.get("title") or ""
    m["_frequency"] = series.get("frequency")
    m["_target_date"] = market_target_date(m)
    if not m.get("title"):
        base = m.get("_event_title") or m["_series_title"] or m.get("event_ticker") or ticker
        if not m.get("_event_title") and m["_target_date"]:
            base = f"{base} on {m['_target_date']}"
        m["title"] = f"{base}: {ysub}" if ysub else base
        m["_title_synthetic"] = True
    if not m.get("subtitle"):
        m["subtitle"] = ysub
    m["_volume"] = _f(m.get("volume_fp")) if m.get("volume_fp") is not None else _f(m.get("volume"))
    m["_open_interest"] = _f(m.get("open_interest_fp")) if m.get("open_interest_fp") is not None else _f(m.get("open_interest"))
    if m.get("volume") is None:
        m["volume"] = m["_volume"]
    m["_weather_kind"] = classify_market(m)
    city = map_city(m)
    m["_city"] = city
    m["_city_name"] = city[0] if city else None
    text_b, strike_b = _bracket_from_text(m), _bracket_from_strikes(m)
    m["_bracket"] = text_b or strike_b
    m["_bracket_conflict"] = _bracket_conflict(text_b, strike_b)
    ya, yb, na, nb = best_side_prices(m)
    m["_prices"] = {"yes_ask": ya, "yes_bid": yb, "no_ask": na, "no_bid": nb}
    if m["_weather_kind"] not in MODELED_KINDS: reason = "unsupported contract type"
    elif m["_target_date"] is None: reason = "not a single-day contract"
    elif city is None: reason = "no city/station mapping"
    elif m["_bracket"] is None: reason = "bracket not parseable"
    else: reason = None
    m["_skip_reason"] = reason
    return m


# --------------------------------------------------------------------- discovery
def _discover_series(client, diag):
    """Return list of weather series dicts. Records outcome in diag['series_discovery']."""
    info = {"ok": False, "count": 0, "error": None, "method": None}
    diag["series_discovery"] = info
    attempts = (("category filter", {"category": WEATHER_CATEGORY}), ("unfiltered + local filter", {}))
    for method, params in attempts:
        try:
            data = client.get("/series", params)
        except ApiError as e:
            info["error"] = f"{method}: {e}"
            continue
        rows = data.get("series") or []
        if method != "category filter":
            rows = [s for s in rows if "weather" in str(s.get("category", "")).lower()
                    or str(s.get("ticker", "")).upper().startswith(KNOWN_PREFIXES)]
        rows = [s for s in rows if s.get("ticker")]
        if rows:
            info.update(ok=True, count=len(rows), method=method, error=None)
            return rows
        info["error"] = f"{method}: returned 0 series"
    return []


def _probe_series():
    prefixes = ("KXHIGH", "KXHIGHT", "KXLOW", "KXLOWT", "KXRAIN", "KXSNOW")
    return [{"ticker": p + code, "title": f"(probe) {p}{code}", "probe": True}
            for code in CITIES for p in prefixes]


def _query_series(client, s, max_pages):
    tk = s["ticker"]
    rec = {"series": tk, "title": s.get("title"), "frequency": s.get("frequency"), "pages": 0,
           "markets": 0, "ok": True, "error": None, "via": "markets", "probe": bool(s.get("probe"))}
    items, pages, trunc, err = client.paginate(
        "/markets", {"series_ticker": tk, "status": "open", "limit": 1000}, "markets", max_pages)
    rec["pages"] = pages
    if err and not items:
        if rec["probe"] and err.status in (400, 404):
            rec["error"] = None  # a guessed ticker that does not exist is not a failure
            return rec, []
        ev, ev_pages, trunc, err2 = client.paginate(
            "/events", {"series_ticker": tk, "status": "open", "with_nested_markets": "true", "limit": 200},
            "events", max_pages)
        rec["pages"] += ev_pages
        if ev or not err2:
            rec["via"] = "events (fallback)"
            for e in ev:
                for mk in e.get("markets") or []:
                    mk.setdefault("_event_title", e.get("title"))
                    mk.setdefault("event_ticker", e.get("event_ticker"))
                    items.append(mk)
            err = err2
        else:
            rec.update(ok=False, error=f"markets: {err}; events: {err2}")
            return rec, []
    if err:
        rec["error"] = f"partial: {err}"
    if trunc:
        rec["error"] = (rec["error"] or "") + f" hit page cap ({max_pages})"
    rec["markets"] = len(items)
    return rec, items


def _feed_scan(client, max_pages, diag):
    """Last resort: general feed, combos excluded, strict weather filter."""
    info = {"pages": 0, "retrieved": 0, "matched": 0, "error": None}
    diag["feed_scan"] = info
    items, pages, trunc, err = client.paginate(
        "/markets", {"status": "open", "limit": 1000, "mve_filter": "exclude"}, "markets", max_pages)
    info.update(pages=pages, retrieved=len(items))
    if err:
        info["error"] = str(err)
    hits = [m for m in items if is_weather_candidate(m)]
    info["matched"] = len(hits)
    if trunc:
        diag["warnings"].append(f"Feed scan stopped at the {max_pages}-page cap; more markets exist.")
    return hits


def discover_weather_markets(max_pages=20, include_other=True, max_workers=4, use_feed_fallback=True):
    """Find open Kalshi weather markets. Never raises; inspect result['status'].

    Returns {"markets": [...], "diagnostics": {...}, "status": ...} where status is
      ok | partial | empty_verified | empty_with_errors | failed
    """
    t0 = time.monotonic()
    client = KalshiClient()
    diag = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy": "series", "series_queried": [], "errors": [], "warnings": [],
            "markets_retrieved": 0, "weather_markets": 0, "pages_searched": 0,
            "feed_scan": None, "skip_reasons": {}, "bracket_conflicts": 0,
            "unmapped_series": [], "duplicates_dropped": 0}

    series_list = _discover_series(client, diag)
    if not series_list:
        diag["strategy"] = "probe"
        diag["warnings"].append(
            "Series discovery failed or was empty: probing candidate series tickers built from the "
            "city table instead (guessed tickers that do not exist simply return nothing).")
        if diag["series_discovery"]["error"]:
            diag["errors"].append(f"Series discovery: {diag['series_discovery']['error']}")
        series_list = _probe_series()
    known = {s["ticker"].upper() for s in series_list}
    for seed in SEED_SERIES:
        if seed not in known:
            series_list.append({"ticker": seed, "title": "Highest temperature in NYC (seed)", "frequency": "daily"})

    meta = {s["ticker"].upper(): s for s in series_list}
    raw, records = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_query_series, client, s, max_pages): s for s in series_list}
        for f in as_completed(futs):
            s = futs[f]
            try:
                rec, items = f.result()
            except Exception as e:  # defensive: never lose a series silently
                rec = {"series": s["ticker"], "title": s.get("title"), "pages": 0, "markets": 0,
                       "ok": False, "error": f"unexpected: {type(e).__name__}: {e}", "via": "markets",
                       "probe": bool(s.get("probe")), "frequency": s.get("frequency")}
                items = []
            records.append(rec)
            for m in items:
                m["_discovered_via"] = rec["series"]
                raw.append(m)

    series_ok = sum(1 for r in records if r["ok"])
    for r in records:
        if not r["ok"]:
            diag["errors"].append(f"Series {r['series']}: {r['error']}")
        elif r["error"]:
            diag["warnings"].append(f"Series {r['series']}: {r['error']}")

    if not raw and use_feed_fallback:
        diag["strategy"] += "+feed"
        diag["warnings"].append("No markets from series queries: scanning the general feed (combos excluded).")
        hits = _feed_scan(client, max_pages, diag)
        if diag["feed_scan"]["error"]:
            diag["errors"].append(f"Feed scan: {diag['feed_scan']['error']}")
        elif diag["feed_scan"]["retrieved"]:
            series_ok += 1
        raw.extend(hits)

    markets, seen = [], set()
    for m in raw:
        tk = str(m.get("ticker", "")).upper()
        if not tk or tk in seen:
            diag["duplicates_dropped"] += 1
            continue
        seen.add(tk)
        s = meta.get(_series_of(m), {})
        normalize_market(m, s)
        if not include_other and m["_weather_kind"] == "other_weather":
            continue
        markets.append(m)
    markets.sort(key=lambda m: (m["_series_ticker"], str(m.get("ticker"))))

    unmapped = set()
    for m in markets:
        if m["_skip_reason"]:
            diag["skip_reasons"][m["_skip_reason"]] = diag["skip_reasons"].get(m["_skip_reason"], 0) + 1
        if m["_skip_reason"] == "no city/station mapping":
            unmapped.add(m["_series_ticker"])
        if m["_bracket_conflict"]:
            diag["bracket_conflicts"] += 1
    if diag["bracket_conflicts"]:
        diag["warnings"].append(
            f"{diag['bracket_conflicts']} markets have a label/strike bracket mismatch (label text was used). "
            "Check their rules before trusting model probabilities.")
    diag["unmapped_series"] = sorted(unmapped)

    records.sort(key=lambda r: (-r["markets"], r["series"]))
    diag["series_queried"] = records
    diag["pages_searched"] = sum(r["pages"] for r in records) + ((diag["feed_scan"] or {}).get("pages", 0))
    diag["markets_retrieved"] = len(raw)
    diag["weather_markets"] = len(markets)
    diag["http_requests"] = client.stats["requests"]
    diag["retries"] = client.stats["retries"]
    diag["rate_limited_429"] = client.stats["rate_limited"]
    diag["base_url_used"] = client.base_in_use
    diag["elapsed_s"] = round(time.monotonic() - t0, 1)

    if markets:
        status = "partial" if diag["errors"] else "ok"
    elif series_ok == 0:
        status = "failed"
    else:
        status = "empty_with_errors" if diag["errors"] else "empty_verified"
    return {"markets": markets, "diagnostics": diag, "status": status}


def fetch_open_weather_markets(max_pages=20):
    """Backward-compatible wrapper: returns a list of market dicts.

    Raises MarketDiscoveryError if the API could not be reached at all, so a
    failed request is never presented as "zero markets".
    """
    result = discover_weather_markets(max_pages=max_pages)
    if result["status"] == "failed":
        errs = "; ".join(result["diagnostics"]["errors"][:3]) or "unknown error"
        raise MarketDiscoveryError(f"Kalshi market discovery failed: {errs}")
    return result["markets"]


if __name__ == "__main__":  # python market_engine.py  -> live check from your own computer
    res = discover_weather_markets()
    d = res["diagnostics"]
    print(f"status={res['status']} strategy={d['strategy']} host={d.get('base_url_used')} "
          f"elapsed={d['elapsed_s']}s")
    print(f"markets retrieved={d['markets_retrieved']} weather markets={d['weather_markets']} "
          f"pages={d['pages_searched']} requests={d['http_requests']} 429s={d['rate_limited_429']}")
    for e in d["errors"][:10]:
        print("ERROR:", e)
    for r in d["series_queried"][:15]:
        if r["markets"]:
            print(f"  {r['series']:<16}{r['markets']:>4} markets  {r.get('title')}")
    for m in res["markets"][:8]:
        print(m["ticker"], "|", m["title"], "|", m.get("_bracket"), "|", m["_prices"]["yes_ask"], "| vol", m["_volume"])
