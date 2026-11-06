
import math
import re
from datetime import datetime, timezone
from statistics import NormalDist

import pandas as pd
import requests
import streamlit as st

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
NWS_HEADERS = {"User-Agent": "PersonalKalshiWeatherDashboard/0.1 (personal research app)"}

# Initial city lookup. The market's own rules remain authoritative for settlement.
CITY_LOOKUP = {
    "NYC": ("New York", 40.7812, -73.9665),
    "NY": ("New York", 40.7812, -73.9665),
    "CHI": ("Chicago", 41.9950, -87.9336),
    "CHICAGO": ("Chicago", 41.9950, -87.9336),
    "LAX": ("Los Angeles", 33.9425, -118.4081),
    "LA": ("Los Angeles", 33.9425, -118.4081),
    "MIA": ("Miami", 25.7959, -80.2870),
    "PHIL": ("Philadelphia", 39.8721, -75.2411),
    "PHX": ("Phoenix", 33.4342, -112.0116),
    "AUS": ("Austin", 30.1975, -97.6664),
    "HOU": ("Houston", 29.9902, -95.3368),
    "DEN": ("Denver", 39.8561, -104.6737),
    "BOS": ("Boston", 42.3656, -71.0096),
    "ATL": ("Atlanta", 33.6407, -84.4277),
    "SEA": ("Seattle", 47.4502, -122.3088),
    "SFO": ("San Francisco", 37.6213, -122.3790),
    "DFW": ("Dallas", 32.8998, -97.0403),
    "LAS": ("Las Vegas", 36.0840, -115.1537),
    "MSP": ("Minneapolis", 44.8848, -93.2223),
    "DCA": ("Washington DC", 38.8512, -77.0402),
}

def get_json(url, params=None, headers=None, timeout=15):
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_weather_markets():
    """Read public Kalshi markets; no API key is needed for this endpoint."""
    all_markets, cursor = [], None
    for _ in range(5):
        params = {"status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = get_json(f"{KALSHI_BASE}/markets", params=params)
        markets = data.get("markets", [])
        for market in markets:
            ticker = (market.get("ticker") or "").upper()
            title = (market.get("title") or "").lower()
            if ticker.startswith(("KXHIGH", "KXLOW", "KXTEMP", "KXRAIN", "KXSNOW")) or any(
                word in title for word in ("temperature", "highest temp", "lowest temp", "rainfall", "snowfall")
            ):
                all_markets.append(market)
        cursor = data.get("cursor")
        if not cursor:
            break
    return all_markets

def city_for_market(market):
    haystack = " ".join(str(market.get(k, "")) for k in ("ticker", "title", "subtitle")).upper()
    # Prefer longer codes to avoid matching a short suffix first.
    for code in sorted(CITY_LOOKUP, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(code)}(?![A-Z])", haystack) or (
            code in (market.get("ticker") or "").upper()
            and (market.get("ticker") or "").upper().endswith(code)
        ):
            return CITY_LOOKUP[code]
    return None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_nws_forecast(city, lat, lon):
    point = get_json(f"https://api.weather.gov/points/{lat},{lon}", headers=NWS_HEADERS)
    forecast_url = point["properties"]["forecast"]
    forecast = get_json(forecast_url, headers=NWS_HEADERS)
    periods = forecast.get("properties", {}).get("periods", [])
    # Forecast periods alternate day/night. Return the next daytime high and nighttime low.
    result = {"high_f": None, "low_f": None, "updated": forecast.get("properties", {}).get("updated")}
    today = datetime.now().astimezone().date()
    for p in periods:
        try:
            pdate = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00")).astimezone().date()
        except Exception:
            continue
        if pdate < today:
            continue
        if p.get("isDaytime") and result["high_f"] is None:
            result["high_f"] = p.get("temperature")
        if not p.get("isDaytime") and result["low_f"] is None:
            result["low_f"] = p.get("temperature")
        if result["high_f"] is not None and result["low_f"] is not None:
            break
    return result

def price_to_probability(market):
    # Kalshi's current API uses dollar-string fields.
    for field in ("yes_ask_dollars", "yes_bid_dollars", "last_price_dollars"):
        value = market.get(field)
        if value not in (None, ""):
            try:
                v = float(value)
                return v if v <= 1 else v / 100
            except (TypeError, ValueError):
                pass
    # Older schema fallback.
    for field in ("yes_ask", "yes_bid", "last_price"):
        value = market.get(field)
        if value is not None:
            try:
                return float(value) / 100
            except (TypeError, ValueError):
                pass
    return None

def parse_bracket(market):
    text = " ".join(str(market.get(k, "")) for k in ("yes_sub_title", "subtitle", "title"))
    # Examples: "70° to 71°", "70 to 71", "70° or above", "70° or below"
    nums = re.findall(r"(?<!\d)(-?\d{1,3})(?:\.\d+)?\s*°?", text)
    lower_text = text.lower()
    if len(nums) >= 2:
        a, b = float(nums[-2]), float(nums[-1])
        return ("range", min(a, b), max(a, b))
    if len(nums) == 1 and any(x in lower_text for x in ("or above", "or higher", "at least", "or more")):
        return ("above", float(nums[0]), None)
    if len(nums) == 1 and any(x in lower_text for x in ("or below", "or lower", "at most", "or less")):
        return ("below", None, float(nums[0]))
    return None

def modeled_probability(market, forecast_f, sigma_f):
    bracket = parse_bracket(market)
    if bracket is None or forecast_f is None:
        return None
    dist = NormalDist(mu=float(forecast_f), sigma=float(sigma_f))
    kind, lo, hi = bracket
    if kind == "range":
        return max(0.0, min(1.0, dist.cdf(hi + 0.5) - dist.cdf(lo - 0.5)))
    if kind == "above":
        return max(0.0, min(1.0, 1 - dist.cdf(lo - 0.5)))
    return max(0.0, min(1.0, dist.cdf(hi + 0.5)))

def fmt_pct(x):
    return "—" if x is None else f"{100*x:.1f}%"

st.set_page_config(page_title="Kalshi Weather Edge", page_icon="🌦️", layout="wide")
st.title("🌦️ Kalshi Weather Edge")
st.caption("Personal weather-market research dashboard · Read-only MVP")

with st.sidebar:
    st.header("Model settings")
    sigma = st.slider("Forecast uncertainty (°F)", 1.0, 6.0, 2.5, 0.25,
                      help="Assumed standard deviation around the NWS forecast. This is a starting assumption, not a calibrated model.")
    min_edge = st.slider("Minimum estimated edge (percentage points)", 0, 30, 5, 1)
    st.caption("Market data refreshes every 60 seconds. NWS forecasts refresh every 15 minutes.")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    with st.spinner("Loading open Kalshi weather markets..."):
        markets = fetch_weather_markets()
except Exception as e:
    st.error(f"Could not load Kalshi markets: {e}")
    st.info("Check your connection and try refreshing. Public market-data access does not require a Kalshi API key.")
    st.stop()

if not markets:
    st.warning("No open weather markets were found in the first pages of Kalshi's open-market feed.")
    st.stop()

st.metric("Open weather markets found", len(markets))
rows = []
forecast_cache = {}
for m in markets:
    city_info = city_for_market(m)
    ticker = m.get("ticker", "")
    title = m.get("title", ticker)
    forecast = None
    market_type = "high" if ticker.upper().startswith("KXHIGH") or "highest" in title.lower() else (
        "low" if ticker.upper().startswith("KXLOW") or "lowest" in title.lower() else "other"
    )
    if city_info and market_type in ("high", "low"):
        city, lat, lon = city_info
        try:
            if city not in forecast_cache:
                forecast_cache[city] = fetch_nws_forecast(city, lat, lon)
            forecast = forecast_cache[city]
        except Exception:
            forecast = None
    forecast_value = None
    if forecast:
        forecast_value = forecast.get("high_f" if market_type == "high" else "low_f")
    model_p = modeled_probability(m, forecast_value, sigma)
    market_p = price_to_probability(m)
    edge = model_p - market_p if model_p is not None and market_p is not None else None
    rows.append({
        "City": city_info[0] if city_info else "Unmapped",
        "Market": title,
        "Ticker": ticker,
        "Type": market_type.title(),
        "NWS forecast (°F)": forecast_value,
        "YES market price": market_p,
        "Model YES probability": model_p,
        "Estimated edge": edge,
        "Volume": m.get("volume_fp", m.get("volume", "—")),
        "Close": m.get("close_time", "—"),
        "URL": "https://kalshi.com/markets/" + ticker.lower(),
    })

df = pd.DataFrame(rows)
if "Estimated edge" in df:
    df["Estimated edge (pp)"] = df["Estimated edge"].apply(lambda x: x * 100 if pd.notna(x) else None)
    opportunities = df[df["Estimated edge (pp)"] >= min_edge].sort_values("Estimated edge (pp)", ascending=False)
else:
    opportunities = pd.DataFrame()

st.subheader("Potential model-versus-market discrepancies")
st.caption("Positive edge means the model probability is above the current YES price. This is a screening signal, not a guaranteed or executable profit.")
if not opportunities.empty:
    view = opportunities[["City", "Market", "Ticker", "NWS forecast (°F)", "YES market price",
                          "Model YES probability", "Estimated edge (pp)", "Volume", "Close"]].copy()
    view["YES market price"] = view["YES market price"].map(fmt_pct)
    view["Model YES probability"] = view["Model YES probability"].map(fmt_pct)
    view["Estimated edge (pp)"] = view["Estimated edge (pp)"].map(lambda x: f"{x:.1f}")
    st.dataframe(view, use_container_width=True, hide_index=True)
else:
    st.info("No contracts currently meet the selected edge threshold with the available forecast and bracket data.")

st.subheader("All detected weather markets")
all_view = df[["City", "Market", "Ticker", "Type", "NWS forecast (°F)", "YES market price",
               "Model YES probability", "Estimated edge", "Volume", "Close", "URL"]].copy()
all_view["YES market price"] = all_view["YES market price"].map(fmt_pct)
all_view["Model YES probability"] = all_view["Model YES probability"].map(fmt_pct)
all_view["Estimated edge"] = all_view["Estimated edge"].map(lambda x: "—" if pd.isna(x) else f"{100*x:.1f} pp")
st.dataframe(all_view, use_container_width=True, hide_index=True,
             column_config={"URL": st.column_config.LinkColumn("Kalshi market", display_text="Open market")})

st.subheader("How to interpret the estimates")
st.markdown("""
- **YES market price** is a displayed market-price proxy, not necessarily the price you can execute at after spread, depth, and fees.
- **Model YES probability** currently uses a simple normal-distribution approximation around an NWS point forecast.
- **Estimated edge** = model probability − displayed YES price. It does not yet subtract trading fees, slippage, or uncertainty in the model.
- **Unmapped markets** need a verified city/station mapping before the app will calculate a forecast-based probability.
""")

st.warning("Important: Kalshi settles each contract using the source and station in its rules. This MVP uses NWS forecasts as an independent signal and does not yet verify every market's settlement station or calibrate forecast error by city, lead time, and market type. Do not treat its estimates as trade recommendations.")
st.caption("Read-only MVP. It does not connect to your Kalshi account, place orders, or access positions.")
