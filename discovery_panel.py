"""Streamlit helpers: cached market discovery + a diagnostics panel. Read-only."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from market_engine import discover_weather_markets


@st.cache_data(ttl=120, show_spinner=False)
def _cached_discovery(max_pages: int):
    return discover_weather_markets(max_pages=max_pages)


def load_weather_markets(max_pages: int = 20):
    """Cached for 2 minutes, but failed lookups are NOT kept: the next rerun retries."""
    result = _cached_discovery(max_pages)
    if result["status"] in ("failed", "empty_with_errors"):
        _cached_discovery.clear()
    return result


def render_discovery_banner(result):
    """One-line status banner. Call right after loading so failures are never hidden."""
    status = result["status"]
    if status == "failed":
        st.error("Kalshi could not be reached, so the market count is unknown (this is NOT 'zero markets'). "
                 "Details are in the diagnostics below.")
    elif status == "empty_with_errors":
        st.warning("No weather markets were returned, and some API requests failed. The list may be incomplete.")
    elif status == "empty_verified":
        st.info("Kalshi answered every request successfully and currently lists no open weather markets "
                "in the queried series.")
    elif status == "partial":
        st.warning("Markets loaded, but some API requests failed (see the Scanner diagnostics tab). "
                   "The list may be incomplete.")


def render_discovery_diagnostics(result, expanded=False):
    d, status = result["diagnostics"], result["status"]
    c1, c2 = st.columns(2)
    c1.metric("Weather markets identified", d["weather_markets"])
    c2.metric("Markets retrieved (raw)", d["markets_retrieved"])
    c3, c4 = st.columns(2)
    c3.metric("Pages searched", d["pages_searched"])
    c4.metric("Series queried", len(d["series_queried"]))

    with st.expander("Discovery diagnostics", expanded=expanded):
        st.caption(f"Strategy: {d['strategy']} · Host: {d.get('base_url_used')} · {d['elapsed_s']} s · "
                   f"{d.get('http_requests', 0)} requests · {d.get('retries', 0)} retries · "
                   f"{d.get('rate_limited_429', 0)} rate-limit (429) responses · {d['generated_at']} UTC")
        sd = d["series_discovery"]
        st.write(f"**Series list:** {'OK' if sd['ok'] else 'FAILED'} · {sd['count']} series"
                 + (f" · via {sd['method']}" if sd.get("method") else "")
                 + (f" · {sd['error']}" if sd.get("error") else ""))
        if d.get("feed_scan"):
            fs = d["feed_scan"]
            st.write(f"**General-feed fallback:** {fs['pages']} pages, {fs['retrieved']} markets scanned, "
                     f"{fs['matched']} weather matches" + (f" · error: {fs['error']}" if fs["error"] else ""))
        for e in d["errors"][:25]:
            st.error(e)
        for w in d["warnings"][:25]:
            st.warning(w)
        if d["skip_reasons"]:
            st.write("**Markets listed but not modelled (reason: count):**")
            st.json(d["skip_reasons"])
        if d["unmapped_series"]:
            st.write("**Series needing a city/station mapping:** " + ", ".join(d["unmapped_series"]))
        rows = [{"Series": r["series"], "Title": r.get("title"), "Freq": r.get("frequency"),
                 "Markets": r["markets"], "Pages": r["pages"], "Via": r["via"],
                 "OK": r["ok"], "Probe": r["probe"], "Note": r["error"]} for r in d["series_queried"]]
        if rows:
            st.write("**Weather series queried:**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
