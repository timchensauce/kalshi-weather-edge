"""Ensemble + statistical weather probabilities using public Open-Meteo APIs."""
from __future__ import annotations
import math
import requests
import numpy as np
import pandas as pd
from statistics import NormalDist

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _get(url, params):
    r = requests.get(url, params=params, timeout=22, headers={"User-Agent": "KalshiWeatherEdge/0.2"})
    r.raise_for_status()
    return r.json()


def fetch_ensemble(lat, lon, timezone="auto", days=7):
    # Use documented model families; ensemble API can return each member as a column.
    params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,precipitation,rain,snowfall",
              "models": "gfs_seamless,ecmwf_ifs025_ensemble", "temperature_unit": "fahrenheit",
              "precipitation_unit": "inch", "timezone": timezone, "forecast_days": days}
    return _get(ENSEMBLE_URL, params)


def fetch_multi_model(lat, lon, timezone="auto", days=7):
    params = {"latitude": lat, "longitude": lon,
              "hourly": "temperature_2m,precipitation_probability,precipitation,rain,snowfall",
              "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,precipitation_probability_max",
              "temperature_unit": "fahrenheit", "precipitation_unit": "inch", "timezone": timezone,
              "forecast_days": days, "models": "gfs_seamless,ecmwf_ifs04,icon_seamless"}
    return _get(FORECAST_URL, params)


def _member_columns(hourly, prefix):
    cols = [k for k in hourly if k.startswith(prefix) and isinstance(hourly[k], list)]
    return cols


def _daily_aggregate(hourly, key_prefix, dates):
    cols = _member_columns(hourly, key_prefix)
    if not cols: return {}
    times = pd.to_datetime(hourly.get("time", []), errors="coerce")
    day_vals = {}
    for d in dates:
        mask = np.array([str(t.date()) == str(d) if not pd.isna(t) else False for t in times])
        vals = []
        for col in cols:
            arr = np.array(hourly[col], dtype=float)
            chunk = arr[mask]
            chunk = chunk[np.isfinite(chunk)]
            if len(chunk):
                vals.append(float(np.max(chunk)) if "temperature" in key_prefix and "max" in key_prefix else
                             float(np.min(chunk)) if "temperature" in key_prefix and "min" in key_prefix else
                             float(np.sum(chunk)))
        day_vals[str(d)] = vals
    return day_vals


def ensemble_daily_distribution(data, kind, target_date):
    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), errors="coerce")
    date_str = str(target_date)
    mask = np.array([str(t.date()) == date_str if not pd.isna(t) else False for t in times])
    if kind in ("high_temperature", "temperature"):
        cols = _member_columns(hourly, "temperature_2m")
        agg = np.max
    elif kind == "low_temperature":
        cols = _member_columns(hourly, "temperature_2m")
        agg = np.min
    elif kind == "snow":
        cols = _member_columns(hourly, "snowfall")
        agg = np.sum
    else:
        cols = _member_columns(hourly, "precipitation")
        agg = np.sum
    vals = []
    for col in cols:
        arr = np.array(hourly[col], dtype=float)
        chunk = arr[mask]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk): vals.append(float(agg(chunk)))
    return vals


def model_daily_value(data, kind, target_date):
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if str(target_date) not in dates: return None
    i = dates.index(str(target_date))
    keys = {"high_temperature": "temperature_2m_max", "temperature": "temperature_2m_max",
            "low_temperature": "temperature_2m_min", "snow": "snowfall_sum",
            "precipitation": "precipitation_sum"}
    key = keys.get(kind)
    arr = daily.get(key, []) if key else []
    try: return float(arr[i])
    except (IndexError, TypeError, ValueError): return None


def normal_cdf(x, mu, sigma):
    return NormalDist(mu=float(mu), sigma=max(float(sigma), 0.05)).cdf(float(x))


def bracket_probability(samples, bracket, kind, unit=None, calibration=None, statistical_sigma=2.5):
    """Blend empirical ensemble-member frequency and a bias-adjusted Gaussian estimate."""
    if not bracket: return None, {}
    samples = [float(x) for x in (samples or []) if x is not None and math.isfinite(float(x))]
    if not samples: return None, {"reason": "No usable ensemble members"}
    lo, hi, bkind = bracket.get("lo"), bracket.get("hi"), bracket.get("kind")
    # Contract threshold text without a unit defaults to Fahrenheit for temperature markets.
    if kind in ("high_temperature", "low_temperature", "temperature") and unit == "C":
        lo = lo * 9/5 + 32 if lo is not None else None
        hi = hi * 9/5 + 32 if hi is not None else None
    if kind in ("high_temperature", "low_temperature", "temperature"):
        # Integer station readings/contract brackets use half-degree continuity boundaries.
        inside = lambda x: (x >= lo - .5 if bkind == "above" else x <= hi + .5 if bkind == "below" else lo - .5 <= x <= hi + .5)
    else:
        inside = lambda x: (x >= lo if bkind == "above" else x <= hi if bkind == "below" else lo <= x <= hi)
    p_ensemble = sum(1 for x in samples if inside(x)) / len(samples)
    mu = float(np.mean(samples))
    sigma = max(float(np.std(samples, ddof=1)) if len(samples) > 1 else statistical_sigma, 0.35)
    bias = float((calibration or {}).get("bias", 0.0))
    residual_sigma = float((calibration or {}).get("sigma", statistical_sigma))
    # Correct the ensemble mean using uploaded forecast-vs-observed residuals.
    mu_adj = mu - bias
    sigma_adj = max(math.sqrt(sigma*sigma + residual_sigma*residual_sigma) / math.sqrt(2), 0.5)
    if kind in ("high_temperature", "low_temperature", "temperature"):
        if bkind == "above": p_stat = 1-normal_cdf(lo-.5, mu_adj, sigma_adj)
        elif bkind == "below": p_stat = normal_cdf(hi+.5, mu_adj, sigma_adj)
        else: p_stat = max(0, normal_cdf(hi+.5, mu_adj, sigma_adj)-normal_cdf(lo-.5, mu_adj, sigma_adj))
    else:
        # Non-negative precipitation/snow distribution: Gaussian approximation is
        # conservative around zero; empirical member frequency gets equal blend weight.
        if bkind == "above": p_stat = 1-normal_cdf(lo, mu_adj, max(sigma_adj, .05))
        elif bkind == "below": p_stat = normal_cdf(hi, mu_adj, max(sigma_adj, .05))
        else: p_stat = max(0, normal_cdf(hi, mu_adj, sigma_adj)-normal_cdf(lo, mu_adj, sigma_adj))
    p = min(1.0, max(0.0, .5*p_ensemble + .5*p_stat))
    return p, {"ensemble_probability": p_ensemble, "statistical_probability": p_stat,
               "ensemble_mean": mu, "ensemble_sd": sigma, "adjusted_mean": mu_adj,
               "adjusted_sigma": sigma_adj, "member_count": len(samples)}



def fetch_nws_daily(lat, lon, target_date, kind, timezone="auto"):
    """Fetch NWS point forecast for a second, independent forecast signal."""
    headers = {"User-Agent": "KalshiWeatherEdge/0.2 (personal weather research)", "Accept": "application/geo+json"}
    point = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=15)
    point.raise_for_status()
    forecast_url = point.json()["properties"]["forecast"]
    forecast = requests.get(forecast_url, headers=headers, timeout=15)
    forecast.raise_for_status()
    periods = forecast.json().get("properties", {}).get("periods", [])
    want_day = kind in ("high_temperature", "temperature")
    want_night = kind == "low_temperature"
    for period in periods:
        try:
            local_date = pd.Timestamp(period["startTime"]).date().isoformat()
        except Exception:
            continue
        if local_date != str(target_date):
            continue
        if (want_day and period.get("isDaytime")) or (want_night and not period.get("isDaytime")):
            value = period.get("temperature")
            if value is not None and period.get("temperatureUnit") == "C":
                value = float(value) * 9/5 + 32
            return {"value": float(value) if value is not None else None,
                    "precipitation_probability": period.get("probabilityOfPrecipitation", {}).get("value"),
                    "short_forecast": period.get("shortForecast"), "source": "NWS point forecast"}
    return {"value": None, "precipitation_probability": None, "short_forecast": None, "source": "NWS point forecast"}


def calibration_from_csv(uploaded):
    """CSV columns: forecast, actual; optional kind column to filter in UI."""
    df = pd.read_csv(uploaded)
    if not {"forecast", "actual"}.issubset(df.columns):
        raise ValueError("Calibration CSV must contain 'forecast' and 'actual' columns.")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna(subset=["forecast", "actual"])
    if len(df) < 10:
        raise ValueError("Please provide at least 10 valid forecast/actual pairs for calibration.")
    residual = df["forecast"] - df["actual"]
    return {"bias": float(residual.mean()), "sigma": max(float(residual.std(ddof=1)), .5), "n": int(len(df))}


def historical_climatology(lat, lon, timezone="auto", days_back=365):
    """Fetch observed historical daily stats as contextual climatology, not forecast skill."""
    from datetime import date, timedelta
    end = date.today() - timedelta(days=5)
    start = end - timedelta(days=days_back)
    params = {"latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
              "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum",
              "temperature_unit": "fahrenheit", "precipitation_unit": "inch", "timezone": timezone}
    return _get(ARCHIVE_URL, params)
