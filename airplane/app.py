import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Airline Route Demand Forecaster", layout="wide")
st.title("✈️ Airline Route Demand Forecasting")
st.caption("SARIMAX(0,1,1)(0,1,1,12) — trained live per selected route on US DOT T-100 data (2010-2024)")

# ---------------------------------------------------------
# LOAD DATA (cached so it only loads once, not on every click)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("route_monthly_data.csv")
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
    return df

df = load_data()

# ---------------------------------------------------------
# SIDEBAR: ROUTE SELECTION
# ---------------------------------------------------------
st.sidebar.header("Select Route")

# Origin dropdown (includes "ALL" for total US domestic view)
origins = sorted(df['ORIGIN'].unique())
selected_origin = st.sidebar.selectbox("Origin", origins, index=origins.index('ALL') if 'ALL' in origins else 0)

# Destination dropdown — filtered based on selected origin
if selected_origin == 'ALL':
    destinations = ['ALL']
else:
    destinations = sorted(df[df['ORIGIN'] == selected_origin]['DEST'].unique())

selected_dest = st.sidebar.selectbox("Destination", destinations)

# Forecast horizon
horizon = st.sidebar.slider("Forecast horizon (months)", min_value=3, max_value=24, value=12, step=3)

run_forecast = st.sidebar.button("Generate Forecast", type="primary")

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
def get_route_series(df, origin, dest):
    route_df = df[(df['ORIGIN'] == origin) & (df['DEST'] == dest)].copy()
    route_df = route_df.sort_values('DATE')
    route_df = route_df.set_index('DATE')
    return route_df['PASSENGERS']

if run_forecast:
    series = get_route_series(df, selected_origin, selected_dest)

    if len(series) < 24:
        st.error(
            f"Not enough historical data for {selected_origin} → {selected_dest} "
            f"({len(series)} months available). Need at least 24 months to fit a seasonal model. "
            "Please pick a different route."
        )
    else:
        with st.spinner(f"Fitting SARIMAX model for {selected_origin} → {selected_dest}..."):
            try:
                model = pm.ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
                model.fit(series)

                forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)

                # -------------------------------------------------
                # PLOT
                # -------------------------------------------------
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(series.index, series.values, label="Actual", color="#1f77b4")
                ax.plot(forecast.index, forecast.values, label="Forecast", color="#d62728")
                ax.fill_between(
                    forecast.index, conf_int[:, 0], conf_int[:, 1],
                    color="#d62728", alpha=0.2, label="95% Confidence Interval"
                )
                ax.set_title(f"Monthly Passenger Demand: {selected_origin} → {selected_dest}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Passengers")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # -------------------------------------------------
                # FORECAST TABLE
                # -------------------------------------------------
                st.subheader("Forecast Table")
                forecast_table = pd.DataFrame({
                    "Date": forecast.index.strftime("%Y-%m"),
                    "Predicted Passengers": forecast.values.round(0).astype(int),
                    "Lower Bound": conf_int[:, 0].round(0).astype(int),
                    "Upper Bound": conf_int[:, 1].round(0).astype(int),
                })
                st.dataframe(forecast_table, use_container_width=True, hide_index=True)

                # -------------------------------------------------
                # QUICK STATS
                # -------------------------------------------------
                col1, col2, col3 = st.columns(3)
                col1.metric("Historical months of data", len(series))
                col2.metric("Avg monthly passengers (last 12mo)", f"{int(series.tail(12).mean()):,}")
                col3.metric("Forecast avg (next period)", f"{int(forecast.values.mean()):,}")

            except Exception as e:
                st.error(f"Model failed to fit for this route: {e}")
else:
    st.info("Select a route from the sidebar and click **Generate Forecast** to begin.")

st.markdown("---")
st.caption(
    "Data source: US DOT Bureau of Transportation Statistics — T-100 Domestic Market data (2010-2024). "
    "Model validated on JFK→LAX with 2.69% MAPE on held-out 2024 data before generalizing to other routes."
)
