# Kalshi Weather Edge — v0.2

Personal, read-only weather-market research dashboard. The original MVP is preserved as `app_v1.py`; the new multi-model version runs from `app.py`.

## What's new

- Broader open-market discovery across multiple Kalshi market-feed pages.
- Weather-market classification for high/low temperature, general temperature, precipitation/rain, and snow contracts.
- Expanded airport/city mapping, including Raleigh (RDU), with station mappings treated as provisional.
- Open-Meteo ensemble member distributions plus deterministic multi-model guidance.
- NWS point forecasts as a separate signal for high/low temperature markets.
- Optional forecast-verification CSV upload (`forecast`, `actual`) to estimate bias and residual uncertainty.
- YES and NO side comparison using available asks (and complementary bid-derived estimates where necessary).
- Configurable fee coefficient, slippage reserve, minimum net EV, and minimum volume.
- In-app alert snapshots, CSV exports, and optional ntfy push alerts that only send after a manual click.
- Contract-rules inspection panel.
- No automated trading or authenticated account access.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## iPhone use

1. Deploy this Streamlit project to a private HTTPS host (for example, a private Streamlit Community Cloud app or your own server).
2. Open the app's HTTPS URL in Safari on iPhone.
3. Use Share → Add to Home Screen.

The dashboard is responsive, but it is a web app shortcut rather than a native iOS binary. Do not make the deployment public if you later add personal API keys or private alert destinations.

## Data sources

- Kalshi public Trade API v2 market data: `https://external-api.kalshi.com/trade-api/v2`
- Open-Meteo Ensemble API: `https://ensemble-api.open-meteo.com/v1/ensemble`
- Open-Meteo Forecast API: `https://api.open-meteo.com/v1/forecast`
- NWS point and forecast API: `https://api.weather.gov`
- Open-Meteo Archive API is included as a helper for historical observed context; observed climatology is not a substitute for forecast-verification data.

Public read-only market data does not require a Kalshi API key. See the current API documentation and terms for usage limits and attribution.

## Probability and EV methodology

1. Retrieve daily ensemble-member values for the contract date and market type.
2. Blend ensemble-member bracket frequency with a bias-adjusted statistical probability.
3. Shift the ensemble distribution toward the consensus of deterministic model forecasts and, for temperature markets, the NWS point forecast.
4. If a verification CSV is uploaded, use its forecast-minus-actual bias and residual standard deviation as a calibration input.
5. Estimate YES and NO expected value per contract from model probability, current ask price, a configurable fee approximation, and a configurable slippage reserve.

This is a research heuristic, not a validated trading model. It is not a full historical backtest, and the optional calibration CSV is supplied by the user. The default fee coefficient is an editable approximation, not a live Kalshi fee quote. Confirm current contract-specific fees and executable prices before trading.

## Important limitations

- Market discovery is paginated and capped at 20 pages per refresh. It may not include every weather contract on the exchange.
- City/station mappings are heuristic. A contract may settle against a particular airport, observation site, weather provider, or report that differs from the mapped forecast grid point.
- Bracket parsing is heuristic. Unsupported or ambiguous contracts are left without model probabilities.
- The current blend uses ensemble member distributions and deterministic model consensus; forecast error calibration is only as good as the uploaded verification data.
- NWS is currently used as an additional point forecast for temperature markets, not as the settlement source.
- A complementary bid-derived ask is only an approximation if the explicit opposite-side ask is absent.
- Displayed prices do not guarantee executable fills. Order-book depth, spread, market impact, liquidity, fee overrides, and slippage may materially change expected value.
- In-app alert snapshots are session-only. ntfy push alerts require a user-supplied topic and a manual send click; no background monitoring is configured.

## Test the core parser and probability engine

```bash
python -m unittest discover -s tests
```
