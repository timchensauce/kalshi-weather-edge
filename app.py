from __future__ import annotations
import os
from datetime import date, datetime
import pandas as pd
import requests
import streamlit as st

from market_engine import (fetch_open_weather_markets, map_city, market_bracket,
    market_target_date, best_side_prices, trade_metrics, classify_market)
from weather_engine import (fetch_ensemble, fetch_multi_model, ensemble_daily_distribution,
    model_daily_value, bracket_probability, calibration_from_csv, historical_climatology, fetch_nws_daily)

st.set_page_config(page_title="Kalshi Weather Edge", page_icon="🌦️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1500px;}
[data-testid="stMetric"] {background:rgba(120,120,120,.07); padding:12px 14px; border-radius:14px;}
.small-note {font-size:.82rem; opacity:.75;}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Kalshi Weather Edge")
st.caption("Personal weather-market intelligence · multi-model probabilities · fee-aware alerts · read-only")

with st.sidebar:
    st.header("Scanner settings")
    min_edge_pp = st.slider("Minimum net EV (¢ per contract)", 0, 30, 5, 1)
    min_volume = st.number_input("Minimum market volume", min_value=0, value=0, step=10)
    fee_rate = st.slider("Estimated fee coefficient", 0.00, 0.15, 0.07, 0.005,
                         help="Approximation only. Confirm the applicable Kalshi fee schedule for each contract.")
    slippage_cents = st.slider("Slippage reserve (¢ / contract)", 0, 10, 1, 1)
    days_ahead = st.slider("Forecast horizon (days)", 1, 7, 5)
    st.divider()
    st.caption("Data refresh")
    if st.button("Refresh markets + forecasts", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

calibration = {"bias": 0.0, "sigma": 2.5, "n": 0}
with st.expander("Historical calibration (optional)", expanded=False):
    st.write("Upload your own historical forecast verification CSV to bias-correct the statistical component. Required columns: `forecast`, `actual`; optional `kind` column. Use same units as the relevant contract (°F for temperature, inches for precipitation).")
    calibration_file = st.file_uploader("Upload calibration CSV", type=["csv"], key="calibration_csv")
    if calibration_file:
        try:
            calibration = calibration_from_csv(calibration_file)
            st.success(f"Calibration loaded: {calibration['n']} pairs · forecast bias {calibration['bias']:+.2f} · residual SD {calibration['sigma']:.2f}")
        except Exception as exc:
            st.error(str(exc))

@st.cache_data(ttl=60, show_spinner=False)
def load_markets():
    return fetch_open_weather_markets()

@st.cache_data(ttl=900, show_spinner=False)
def load_city_weather(city, lat, lon, tz, horizon):
    ens = fetch_ensemble(lat, lon, tz, horizon)
    multi = fetch_multi_model(lat, lon, tz, horizon)
    return ens, multi

try:
    with st.spinner("Scanning open Kalshi weather markets..."):
        markets = load_markets()
except Exception as exc:
    st.error(f"Kalshi market feed could not be loaded: {exc}")
    st.info("This app uses public market-data endpoints and does not need an API key for read-only scanning.")
    st.stop()

if not markets:
    st.warning("No open weather markets were detected in the current market-feed pages.")
    st.stop()

rows, city_weather_cache, model_errors = [], {}, []
progress = st.progress(0, text="Evaluating market contracts...")
for idx, market in enumerate(markets):
    kind = market.get("_weather_kind") or classify_market(market)
    city_info = map_city(market)
    target = market_target_date(market)
    bracket = market_bracket(market)
    forecast_meta, probability, model_details = {}, None, {}
    if city_info and target and bracket and kind in ("high_temperature", "low_temperature", "temperature", "precipitation", "snow"):
        city, lat, lon, tz = city_info
        try:
            if city not in city_weather_cache:
                city_weather_cache[city] = load_city_weather(city, lat, lon, tz, days_ahead)
            ensemble_data, multi_data = city_weather_cache[city]
            samples = ensemble_daily_distribution(ensemble_data, kind, target)
            # Statistical input: consensus daily point estimate across available deterministic models.
            # Open-Meteo may suffix daily fields by model; collect all matching arrays.
            daily = multi_data.get("daily", {})
            dates = daily.get("time", [])
            di = dates.index(target) if target in dates else None
            key_map = {"high_temperature":"temperature_2m_max", "temperature":"temperature_2m_max",
                       "low_temperature":"temperature_2m_min", "precipitation":"precipitation_sum", "snow":"snowfall_sum"}
            prefix = key_map[kind]
            deterministic = []
            if di is not None:
                for key, vals in daily.items():
                    if key.startswith(prefix) and isinstance(vals, list) and di < len(vals) and vals[di] is not None:
                        try: deterministic.append(float(vals[di]))
                        except (ValueError, TypeError): pass
            independent_points = list(deterministic)
            if kind in ("high_temperature", "low_temperature", "temperature"):
                try:
                    nws = fetch_nws_daily(lat, lon, target, kind, tz)
                    if nws.get("value") is not None:
                        independent_points.append(nws["value"])
                        model_details["nws_point_forecast"] = nws["value"]
                except Exception as nws_exc:
                    model_details["nws_warning"] = str(nws_exc)
            if independent_points and samples:
                # Shift ensemble center toward the consensus of independent deterministic guidance.
                shift = sum(independent_points)/len(independent_points) - (sum(samples)/len(samples))
                samples = [x + shift for x in samples]
                model_details["independent_model_count"] = len(independent_points)
                model_details["independent_consensus"] = sum(independent_points)/len(independent_points)
            probability, stats = bracket_probability(samples, bracket, kind, bracket.get("unit"), calibration)
            model_details.update(stats)
            forecast_meta = {"city": city, "date": target, "members": len(samples)}
        except Exception as exc:
            model_errors.append(f"{city}: {exc}")
    yes_ask, yes_bid, no_ask, no_bid = best_side_prices(market)
    metrics = trade_metrics(probability, yes_ask, no_ask, fee_rate)
    candidates = []
    for side, values in metrics.items():
        net_ev = values["ev_per_contract"] - slippage_cents/100
        candidates.append((side, values, net_ev))
    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0] if candidates else (None, {}, None)
    side, side_values, net_ev = best
    volume = market.get("volume_fp", market.get("volume", 0))
    try: volume_num = float(volume or 0)
    except Exception: volume_num = 0
    spread = (yes_ask-yes_bid)*100 if yes_ask is not None and yes_bid is not None else None
    rows.append({
        "City": forecast_meta.get("city", city_info[0] if city_info else "Unmapped"),
        "Date": target or "—", "Market type": kind.replace("_", " ").title(),
        "Contract": market.get("title") or market.get("ticker"), "Ticker": market.get("ticker"),
        "Bracket": f"{bracket}" if bracket else "Unparsed",
        "YES ask": yes_ask, "NO ask": no_ask, "YES bid": yes_bid,
        "YES model probability": probability, "Ensemble members": model_details.get("member_count", 0),
        "Model consensus": model_details.get("independent_consensus"),
        "NWS forecast": model_details.get("nws_point_forecast"),
        "Best side": side, "Entry price": side_values.get("entry_price"),
        "Estimated fee": side_values.get("fee_estimate"), "Net EV / contract": net_ev,
        "Net EV (¢)": net_ev*100 if net_ev is not None else None,
        "Spread (¢)": spread, "Volume": volume_num,
        "Rules": market.get("rules_primary") or market.get("rules_secondary") or "Review contract rules",
        "URL": f"https://kalshi.com/markets/{str(market.get('ticker','')).lower()}"
    })
    progress.progress((idx+1)/len(markets), text=f"Evaluating market contracts... {idx+1}/{len(markets)}")
progress.empty()

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No weather markets returned.")
    st.stop()

valid = df[df["Net EV (¢)"].notna() & (df["Volume"] >= min_volume)].copy()
opps = valid[valid["Net EV (¢)"] >= min_edge_pp].sort_values("Net EV (¢)", ascending=False)

m1,m2,m3,m4 = st.columns(4)
m1.metric("Weather markets", f"{len(df):,}")
m2.metric("Markets with model", f"{df['YES model probability'].notna().sum():,}")
m3.metric("Alert candidates", f"{len(opps):,}")
m4.metric("Mapped cities", f"{df.loc[df.City != 'Unmapped','City'].nunique():,}")

scan_tab, all_tab, calibration_tab, methodology_tab = st.tabs(["🚨 Edge & alerts", "All markets", "Model diagnostics", "Methodology"])
with scan_tab:
    st.subheader("Potential trades after estimated costs")
    st.caption("A positive net EV means estimated model payout exceeds entry price, estimated fee, and your slippage reserve. It is not a guarantee or a trade instruction.")
    if opps.empty:
        st.info("No mapped, parsed contracts currently clear your net-EV threshold. Try lowering the threshold or checking again later.")
    else:
        display = opps[["City","Date","Market type","Contract","Ticker","Best side","Entry price","YES model probability","Net EV (¢)","Spread (¢)","Volume","URL"]].copy()
        for c in ("Entry price","YES model probability"):
            display[c] = display[c].map(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%")
        display["Net EV (¢)"] = display["Net EV (¢)"].map(lambda x:f"{x:.2f}¢")
        display["Spread (¢)"] = display["Spread (¢)"].map(lambda x:"—" if pd.isna(x) else f"{x:.1f}¢")
        st.dataframe(display, use_container_width=True, hide_index=True,
            column_config={"URL":st.column_config.LinkColumn("Market",display_text="Open")})
        st.download_button("Download opportunity report (CSV)", opps.to_csv(index=False).encode(), "kalshi_weather_opportunities.csv", "text/csv")
    st.subheader("Alert rules")
    st.write(f"Current in-app alert threshold: net EV ≥ {min_edge_pp}¢ per contract, market volume ≥ {min_volume}.")
    if st.button("Create in-app alert snapshot", type="primary", disabled=opps.empty):
        st.session_state["alert_snapshot"] = opps[["City","Date","Contract","Ticker","Best side","Net EV (¢)","URL"]].copy()
        st.success(f"Saved {len(opps)} qualifying signals in this session.")
    if "alert_snapshot" in st.session_state:
        st.dataframe(st.session_state["alert_snapshot"], use_container_width=True, hide_index=True,
            column_config={"URL":st.column_config.LinkColumn("Market",display_text="Open")})
    ntfy_topic = st.text_input("Optional ntfy topic for push alerts", type="password", help="Create a private topic at ntfy.sh and enter it here. This app only sends when you press the button.")
    if st.button("Send current qualifying alerts to ntfy", disabled=opps.empty or not ntfy_topic.strip()):
        try:
            lines = [f"{r['City']} · {r['Date']} · {r['Best side']} · net EV {r['Net EV (¢)']:.1f}¢ · {r['Contract']}" for _,r in opps.head(20).iterrows()]
            resp = requests.post(f"https://ntfy.sh/{ntfy_topic.strip()}", data=("\n".join(lines)).encode(), headers={"Title":"Kalshi weather opportunities","Priority":"default"}, timeout=12)
            resp.raise_for_status()
            st.success(f"Sent {min(len(opps),20)} alert lines to your ntfy topic.")
        except Exception as exc:
            st.error(f"Push alert failed: {exc}")

with all_tab:
    st.subheader("All detected weather contracts")
    filters = st.multiselect("Market types", sorted(df["Market type"].unique()), default=sorted(df["Market type"].unique()))
    view = df[df["Market type"].isin(filters)].copy()
    show = view[["City","Date","Market type","Contract","Ticker","YES ask","NO ask","YES model probability","Best side","Net EV (¢)","Spread (¢)","Volume","URL"]].copy()
    for c in ("YES ask","NO ask","YES model probability"):
        show[c] = show[c].map(lambda x:"—" if pd.isna(x) else f"{x*100:.1f}%")
    show["Net EV (¢)"] = show["Net EV (¢)"].map(lambda x:"—" if pd.isna(x) else f"{x:.2f}¢")
    show["Spread (¢)"] = show["Spread (¢)"].map(lambda x:"—" if pd.isna(x) else f"{x:.1f}¢")
    st.dataframe(show, use_container_width=True, hide_index=True,
        column_config={"URL":st.column_config.LinkColumn("Market",display_text="Open")})
    st.download_button("Export all markets (CSV)", view.to_csv(index=False).encode(), "kalshi_weather_all_markets.csv", "text/csv")
    st.subheader("Contract settlement rules")
    rule_ticker = st.selectbox("Choose a contract to inspect", df["Ticker"].dropna().astype(str).tolist())
    rule_row = df[df["Ticker"] == rule_ticker].iloc[0]
    st.write(f"**{rule_row['Contract']}**")
    st.write(rule_row["Rules"])
    st.caption("Always verify the station, measurement window, and official settlement source in the Kalshi contract itself.")

with calibration_tab:
    st.subheader("Forecast diagnostics")
    if calibration.get("n",0):
        a,b,c = st.columns(3)
        a.metric("Verification pairs", calibration["n"])
        b.metric("Forecast bias", f"{calibration['bias']:+.2f}")
        c.metric("Residual standard deviation", f"{calibration['sigma']:.2f}")
    else:
        st.info("No historical verification CSV uploaded. The model uses a conservative default statistical spread; probabilities are not yet empirically calibrated.")
    st.write("The probability engine blends ensemble-member frequencies with a statistical distribution. Deterministic model output shifts the ensemble center when available. Uploaded verification pairs adjust bias and residual uncertainty.")
    if "YES model probability" in df:
        chart = df.dropna(subset=["YES model probability"])[["Contract","YES model probability"]].head(30).set_index("Contract")
        if not chart.empty: st.bar_chart(chart)
    st.caption("Historical observed climatology is not the same as historical forecast verification. The calibration upload is the supported path for correcting forecast bias and error spread in this version.")

with methodology_tab:
    st.subheader("How the model works")
    st.markdown("""
**Forecast blend**
1. Open-Meteo ensemble members provide a distribution of plausible daily outcomes.
2. Independent Open-Meteo deterministic model forecasts and the NWS point forecast provide a consensus shift.
3. The ensemble probability is blended 50/50 with a bias-adjusted statistical probability.
4. Optional user-provided historical forecast/actual pairs estimate systematic bias and residual spread.

**Trade value**
- YES EV = model P(YES) − YES ask − estimated fee − slippage reserve.
- NO EV = model P(NO) − NO ask − estimated fee − slippage reserve.
- The app ranks the better side when both ask prices are available.

**Alerting**
- In-app alert snapshots are session-only.
- Optional ntfy push alerts are sent only when you press the send button.
- No scheduled background monitoring, account connection, or order placement is enabled.
""")
    st.warning("Settlement-source caution: the city/airport mapping is only a starting proxy. A Kalshi market may settle against a specific station, observation time, weather provider, or report. Inspect each contract's rules and settlement source before acting. The market scanner intentionally leaves unknown or unparsed contracts without a probability.")
    st.warning("Fee caution: the fee coefficient is an editable estimate, not a live Kalshi fee quote. Check current contract-specific fees, actual executable ask, order-book depth, and slippage before trading.")

if model_errors:
    with st.expander(f"Data warnings ({len(model_errors)})"):
        for e in sorted(set(model_errors))[:20]: st.write("• " + e)

st.divider()
st.caption("Kalshi Weather Edge v0.2 · read-only research tool · Open-Meteo ensemble and forecast APIs · Kalshi public market data")
