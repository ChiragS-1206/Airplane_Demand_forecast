# ✈️ Airline Route Demand Forecasting

A time-series forecasting tool that predicts monthly passenger demand for any US domestic airline route, built on 15 years (2010–2024) of official US DOT T-100 data.

**🔗 Live app:** _add your Streamlit Cloud link here after deploying_

---

## Overview

This project forecasts monthly airline passenger demand using a tuned **SARIMAX** time-series model. It started as a deep-dive on a single route (JFK → LAX) and was generalized into an interactive tool that works for any of ~3,700 active US domestic routes, plus a nationwide "ALL routes" view.

## Key Features

- **Live per-route forecasting** — pick any Origin/Destination pair, model fits on-the-fly
- **Adjustable forecast horizon** — 3 to 24 months ahead
- **Confidence intervals** — every forecast includes upper/lower bounds, not just a point estimate
- **Validated accuracy** — final model achieves **2.69% MAPE** on held-out 2024 data (JFK-LAX)

## Data

- **Source:** [US DOT Bureau of Transportation Statistics — T-100 Domestic Market data](https://www.transtats.bts.gov)
- **Range:** January 2010 – December 2024, monthly granularity
- **Fields used:** Year, Month, Origin, Destination, Unique Carrier, Passengers
- Raw data was aggregated to monthly totals per route, filtered to ~3,700 routes with meaningful traffic (>500,000 total passengers over the period), and combined with a nationwide "ALL" aggregate.

## Modeling Approach

Several approaches were tested and compared on a JFK→LAX train/test split (train: 2010–2023, test: 2024):

| Model | MAPE |
|---|---|
| SARIMA (auto_arima default order) | 5.68% |
| Prophet (tuned with COVID regressor) | 10.22% |
| SARIMAX + COVID exogenous flag | 3.67%–4.10% |
| **SARIMAX(0,1,1)(0,1,1,12) — final** | **2.69%** ✅ |

The final model was selected after:
- ADF stationarity testing (confirmed 1 order of differencing needed)
- ACF/PACF analysis to identify seasonal structure (12-month cycle)
- Manual and automated (`auto_arima`) order search
- Residual diagnostics (Ljung-Box test) to confirm no significant leftover autocorrelation
- Head-to-head comparison against Facebook Prophet

Prophet underperformed on this dataset, likely because its flexible trend model is better suited to longer/more complex series, while SARIMAX's explicit differencing better captured the sharp COVID-19 shock and recovery pattern in this ~14-year monthly series.

## Project Structure

```
├── app.py                    # Streamlit app (deployment entry point)
├── prepare_data.py            # One-time script: raw data → route_monthly_data.csv
├── route_monthly_data.csv     # Aggregated, filtered dataset used by the app
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone https://github.com/<your-username>/airline-demand-forecast.git
cd airline-demand-forecast
pip install -r requirements.txt
streamlit run app.py
```

## Regenerating the Dataset

If you want to rebuild `route_monthly_data.csv` from raw T-100 data:

1. Download T-100 Domestic Market/Segment data from [transtats.bts.gov](https://www.transtats.bts.gov)
2. Save it as `final_airplane.csv` in the project root
3. Run:
```bash
python prepare_data.py
```

## Tech Stack

- **pmdarima / statsmodels** — SARIMAX modeling
- **Prophet** — comparison model
- **pandas / numpy** — data processing
- **Streamlit** — deployment
- **matplotlib** — visualization

## Limitations & Future Work

- The SARIMAX order `(0,1,1)(0,1,1,12)` was tuned on JFK-LAX specifically and applied uniformly to all routes; smaller/newer routes may benefit from route-specific tuning
- Model does not currently account for external factors like fuel prices or major world events beyond COVID
- Live per-route fitting takes a few seconds — acceptable for a demo, but a production system would pre-compute or cache models
