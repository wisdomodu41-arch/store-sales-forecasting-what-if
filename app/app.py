import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Store Sales Forecasting Analytics",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Store Sales Forecasting & What-If Analytics")
st.markdown(
    """
    **Business Forecasting Application**

    This application uses a machine-learning forecasting pipeline to
    analyze historical store sales, evaluate prediction performance,
    generate future sales forecasts, and perform what-if scenario analysis.
    """
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).parent


def load_csv(filename):
    path = BASE_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return None


forecast = load_csv("sales_forecast.csv")
validation = load_csv("forecast_validation.csv")
metrics = load_csv("forecast_metrics.csv")
feature_importance = load_csv("feature_importance.csv")
daily_sales = load_csv("daily_sales_analysis.csv")


# ============================================================
# CHECK DATA
# ============================================================

if forecast is None:
    st.error(
        "sales_forecast.csv was not found. "
        "Make sure app.py is in the same folder as the forecasting CSV files."
    )
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """
    Find a column using several possible names.
    """
    if df is None:
        return None

    lower_columns = {col.lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in lower_columns:
            return lower_columns[name.lower()]

    return None


# Find forecast date column
forecast_date_col = find_column(
    forecast,
    ["date", "ds", "forecast_date", "timestamp"]
)
# ============================================================
# FORECAST COLUMN
# ============================================================

forecast_value_col = "forecast_sales"

if forecast_value_col not in forecast.columns:
    st.error(
        "forecast_sales column was not found in sales_forecast.csv."
    )
    st.stop()


if forecast_date_col is not None:
    forecast[forecast_date_col] = pd.to_datetime(
        forecast[forecast_date_col],
        errors="coerce"
    )


# ============================================================
# FORECAST DATA
# ============================================================

forecast_data = forecast.copy()

if forecast_date_col is not None:
    forecast_data = forecast_data.sort_values(
        forecast_date_col
    )


forecast_values = pd.to_numeric(
    forecast_data[forecast_value_col],
    errors="coerce"
).fillna(0)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Forecast Controls")

max_horizon = len(forecast_data)

default_horizon = min(30, max_horizon)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    min_value=7 if max_horizon >= 7 else 1,
    max_value=max_horizon,
    value=default_horizon,
    step=1
)

growth_rate = st.sidebar.slider(
    "What-If Growth Rate",
    min_value=-30,
    max_value=30,
    value=0,
    step=1,
    format="%d%%"
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Scenario interpretation**

    0% = Base forecast

    Positive values = Upside scenario

    Negative values = Downside scenario
    """
)


# ============================================================
# SELECT FORECAST PERIOD
# ============================================================

selected_forecast = forecast_data.iloc[:forecast_horizon].copy()

base_values = pd.to_numeric(
    selected_forecast[forecast_value_col],
    errors="coerce"
).fillna(0)

scenario_multiplier = 1 + (growth_rate / 100)

scenario_values = base_values * scenario_multiplier


# ============================================================
# KEY BUSINESS METRICS
# ============================================================

base_total = base_values.sum()

scenario_total = scenario_values.sum()

difference = scenario_total - base_total

percentage_change = (
    (difference / base_total) * 100
    if base_total != 0
    else 0
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Forecast Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Base Forecast",
        f"{base_total:,.0f}"
    )

with col2:
    st.metric(
        "Scenario Forecast",
        f"{scenario_total:,.0f}",
        f"{percentage_change:+.1f}%"
    )

with col3:
    st.metric(
        "Scenario Impact",
        f"{difference:,.0f}"
    )

with col4:
    st.metric(
        "Forecast Period",
        f"{forecast_horizon} days"
    )


st.divider()


# ============================================================
# FORECAST CHART
# ============================================================

st.subheader("📈 Forecast Trend")

fig, ax = plt.subplots(figsize=(12, 5))

if forecast_date_col is not None:

    dates = selected_forecast[forecast_date_col]

    ax.plot(
        dates,
        base_values,
        label="Base Forecast"
    )

    ax.plot(
        dates,
        scenario_values,
        linestyle="--",
        label=f"What-If Scenario ({growth_rate:+d}%)"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Base Forecast vs What-If Scenario")

else:

    x = range(len(selected_forecast))

    ax.plot(
        x,
        base_values,
        label="Base Forecast"
    )

    ax.plot(
        x,
        scenario_values,
        linestyle="--",
        label=f"What-If Scenario ({growth_rate:+d}%)"
    )

    ax.set_xlabel("Forecast Period")
    ax.set_ylabel("Sales")
    ax.set_title("Base Forecast vs What-If Scenario")


ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# SCENARIO COMPARISON
# ============================================================

st.subheader("🔎 Scenario Analysis")

scenario_data = pd.DataFrame(
    {
        "Scenario": [
            "Pessimistic (-10%)",
            "Base Forecast",
            "Optimistic (+10%)",
            "Selected What-If"
        ],
        "Projected Sales": [
            base_total * 0.90,
            base_total,
            base_total * 1.10,
            scenario_total
        ]
    }
)

st.dataframe(
    scenario_data.style.format(
        {"Projected Sales": "{:,.0f}"}
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

if validation is not None:

    st.divider()

    st.subheader("🎯 Model Validation")

    validation_date_col = find_column(
        validation,
        ["date", "ds", "timestamp"]
    )

    actual_col = find_column(
        validation,
        ["sales", "actual", "actual_sales"]
    )

    predicted_col = find_column(
        validation,
        [
            "predicted_sales",
            "predicted",
            "forecast",
            "prediction"
        ]
    )

    if (
        validation_date_col is not None
        and actual_col is not None
        and predicted_col is not None
    ):

        validation[validation_date_col] = pd.to_datetime(
            validation[validation_date_col],
            errors="coerce"
        )

        validation[actual_col] = pd.to_numeric(
            validation[actual_col],
            errors="coerce"
        )

        validation[predicted_col] = pd.to_numeric(
            validation[predicted_col],
            errors="coerce"
        )

        fig2, ax2 = plt.subplots(figsize=(12, 5))

        ax2.plot(
            validation[validation_date_col],
            validation[actual_col],
            label="Actual Sales"
        )

        ax2.plot(
            validation[validation_date_col],
            validation[predicted_col],
            linestyle="--",
            label="Predicted Sales"
        )

        ax2.set_title(
            "Actual vs Predicted Sales"
        )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Sales")

        ax2.legend()
        ax2.grid(alpha=0.3)

        st.pyplot(fig2)

        plt.close(fig2)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

if metrics is not None:

    st.divider()

    st.subheader("🧠 Model Performance")

    metric_columns = metrics.columns.tolist()

    if len(metric_columns) >= 2:

        metric_name_col = metric_columns[0]
        metric_value_col = metric_columns[1]

        metric_dict = {}

        for _, row in metrics.iterrows():

            name = str(row[metric_name_col]).upper()

            try:
                value = float(row[metric_value_col])
                metric_dict[name] = value
            except:
                pass

        c1, c2, c3 = st.columns(3)

        with c1:
            if "MAE" in metric_dict:
                st.metric(
                    "MAE",
                    f"{metric_dict['MAE']:,.2f}"
                )

        with c2:
            if "RMSE" in metric_dict:
                st.metric(
                    "RMSE",
                    f"{metric_dict['RMSE']:,.2f}"
                )

        with c3:
            if "MAPE" in metric_dict:
                st.metric(
                    "MAPE",
                    f"{metric_dict['MAPE']:.2f}%"
                )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if feature_importance is not None:

    st.divider()

    st.subheader("🔬 Forecasting Drivers")

    feature_columns = feature_importance.columns.tolist()

    if len(feature_columns) >= 2:

        feature_name_col = feature_columns[0]
        importance_col = feature_columns[1]

        importance_data = feature_importance.copy()

        importance_data[importance_col] = pd.to_numeric(
            importance_data[importance_col],
            errors="coerce"
        )

        importance_data = (
            importance_data
            .dropna(subset=[importance_col])
            .sort_values(
                importance_col,
                ascending=False
            )
            .head(10)
        )

        fig3, ax3 = plt.subplots(figsize=(10, 5))

        ax3.barh(
            importance_data[feature_name_col].astype(str),
            importance_data[importance_col]
        )

        ax3.invert_yaxis()

        ax3.set_title(
            "Top Forecasting Features"
        )

        ax3.set_xlabel(
            "Feature Importance"
        )

        st.pyplot(fig3)

        plt.close(fig3)


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.divider()

st.subheader("💼 Business Interpretation")

if growth_rate > 0:

    st.success(
        f"""
        The selected scenario assumes **{growth_rate}% sales growth**
        compared with the base forecast.

        Under this scenario, projected sales increase by approximately
        **{difference:,.0f}** over the selected forecast period.
        """
    )

elif growth_rate < 0:

    st.warning(
        f"""
        The selected scenario assumes a **{abs(growth_rate)}% decline**
        compared with the base forecast.

        Under this scenario, projected sales decrease by approximately
        **{abs(difference):,.0f}** over the selected forecast period.
        """
    )

else:

    st.info(
        """
        The application is currently displaying the base forecast
        without an additional growth adjustment.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Store Sales Forecasting & What-If Analytics | "
    "Python • Machine Learning • Streamlit"
)
