# ============================================================
# STORE SALES FORECASTING & WHAT-IF ANALYTICS
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
stores = pd.read_csv("stores.csv")
oil = pd.read_csv("oil.csv")
holidays = pd.read_csv("holidays_events.csv")
transactions = pd.read_csv("transactions.csv")

print("\nDATASETS LOADED SUCCESSFULLY")

print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)
print("Stores shape:", stores.shape)
print("Oil shape:", oil.shape)
print("Holidays shape:", holidays.shape)
print("Transactions shape:", transactions.shape)


# ============================================================
# 2. CONVERT DATES
# ============================================================

train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
oil["date"] = pd.to_datetime(oil["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
transactions["date"] = pd.to_datetime(transactions["date"])


# ============================================================
# 3. BASIC DATA INSPECTION
# ============================================================

print("\nTRAIN DATA")
print(train.head())

print("\nTRAIN COLUMNS")
print(train.columns.tolist())

print("\nMISSING VALUES")
print(train.isnull().sum())


# ============================================================
# 4. AGGREGATE SALES BY DAY
# ============================================================

daily_sales = (
    train
    .groupby("date", as_index=False)["sales"]
    .sum()
)

daily_sales = daily_sales.sort_values("date")


# ============================================================
# 5. AGGREGATE PROMOTIONS BY DAY
# ============================================================

daily_promotions_train = (
    train
    .groupby("date", as_index=False)["onpromotion"]
    .sum()
)

daily_promotions_test = (
    test
    .groupby("date", as_index=False)["onpromotion"]
    .sum()
)


# ============================================================
# 6. AGGREGATE TRANSACTIONS BY DAY
# ============================================================

# Transactions are retained for descriptive analysis,
# but NOT used as a forecasting feature because future
# transaction volumes are not known in advance.

daily_transactions = (
    transactions
    .groupby("date", as_index=False)["transactions"]
    .sum()
)


# ============================================================
# 7. PREPARE OIL DATA
# ============================================================

daily_oil = oil[
    ["date", "dcoilwtico"]
].copy()

daily_oil = daily_oil.sort_values("date")

daily_oil["dcoilwtico"] = (
    daily_oil["dcoilwtico"]
    .ffill()
    .bfill()
)


# ============================================================
# 8. CREATE HOLIDAY INDICATOR
# ============================================================

holiday_dates = (
    holidays[
        holidays["transferred"] == False
    ]["date"]
    .drop_duplicates()
)

holiday_dates = set(holiday_dates)


# ============================================================
# 9. COMBINE DAILY DATA
# ============================================================

daily = daily_sales.merge(
    daily_promotions_train,
    on="date",
    how="left"
)

daily = daily.merge(
    daily_transactions,
    on="date",
    how="left"
)

daily = daily.merge(
    daily_oil,
    on="date",
    how="left"
)


# ============================================================
# 10. FILL MISSING VALUES
# ============================================================

daily["onpromotion"] = (
    daily["onpromotion"]
    .fillna(0)
)

daily["transactions"] = (
    daily["transactions"]
    .fillna(0)
)

daily["dcoilwtico"] = (
    daily["dcoilwtico"]
    .ffill()
    .bfill()
)


daily["is_holiday"] = (
    daily["date"]
    .isin(holiday_dates)
    .astype(int)
)


# ============================================================
# 11. TIME FEATURES
# ============================================================

daily["day_of_week"] = (
    daily["date"].dt.dayofweek
)

daily["day_of_month"] = (
    daily["date"].dt.day
)

daily["month"] = (
    daily["date"].dt.month
)

daily["quarter"] = (
    daily["date"].dt.quarter
)

daily["year"] = (
    daily["date"].dt.year
)

daily["week_of_year"] = (
    daily["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)


# ============================================================
# 12. LAG FEATURES
# ============================================================

daily["lag_1"] = (
    daily["sales"].shift(1)
)

daily["lag_7"] = (
    daily["sales"].shift(7)
)

daily["lag_14"] = (
    daily["sales"].shift(14)
)

daily["lag_28"] = (
    daily["sales"].shift(28)
)


# ============================================================
# 13. ROLLING FEATURES
# ============================================================

daily["rolling_7"] = (
    daily["sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily["rolling_14"] = (
    daily["sales"]
    .shift(1)
    .rolling(14)
    .mean()
)

daily["rolling_28"] = (
    daily["sales"]
    .shift(1)
    .rolling(28)
    .mean()
)


# ============================================================
# 14. REMOVE EARLY ROWS CREATED BY LAGS
# ============================================================

daily_model = daily.dropna().copy()


# ============================================================
# 15. SELECT FORECASTING FEATURES
# ============================================================

# Transactions are intentionally excluded because
# future transaction volumes are not known.

features = [
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "year",
    "week_of_year",
    "onpromotion",
    "dcoilwtico",
    "is_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_14",
    "rolling_28"
]


X = daily_model[features]

y = daily_model["sales"]


# ============================================================
# 16. TIME-BASED TRAIN / VALIDATION SPLIT
# ============================================================

# No random shuffling.
# The most recent 60 days are reserved for validation.

validation_days = 60

split_date = (
    daily_model["date"].max()
    - pd.Timedelta(days=validation_days)
)

train_data = daily_model[
    daily_model["date"] <= split_date
].copy()

validation_data = daily_model[
    daily_model["date"] > split_date
].copy()


X_train = train_data[features]
y_train = train_data["sales"]

X_valid = validation_data[features]
y_valid = validation_data["sales"]


print("\nTRAINING ROWS:", len(train_data))
print("VALIDATION ROWS:", len(validation_data))


# ============================================================
# 17. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining forecasting model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training complete.")


# ============================================================
# 18. RECURSIVE VALIDATION FORECAST
# ============================================================

print("\nRunning recursive validation...")

validation_history = train_data[
    ["date", "sales"]
].copy()

validation_history = (
    validation_history
    .sort_values("date")
    .reset_index(drop=True)
)


validation_predictions = []


# For validation we only use the latest known oil price
# from the training period, matching the future forecasting
# approach where future oil prices are unknown.

validation_oil = (
    train_data["dcoilwtico"].iloc[-1]
)


for _, validation_row in validation_data.iterrows():

    validation_date = (
        validation_row["date"]
    )


    # --------------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------------

    lag_1 = (
        validation_history["sales"].iloc[-1]
    )

    lag_7 = (
        validation_history["sales"].iloc[-7]
    )

    lag_14 = (
        validation_history["sales"].iloc[-14]
    )

    lag_28 = (
        validation_history["sales"].iloc[-28]
    )


    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    rolling_7 = (
        validation_history["sales"]
        .tail(7)
        .mean()
    )

    rolling_14 = (
        validation_history["sales"]
        .tail(14)
        .mean()
    )

    rolling_28 = (
        validation_history["sales"]
        .tail(28)
        .mean()
    )


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    model_row = pd.DataFrame([{

        "day_of_week":
            validation_date.dayofweek,

        "day_of_month":
            validation_date.day,

        "month":
            validation_date.month,

        "quarter":
            validation_date.quarter,

        "year":
            validation_date.year,

        "week_of_year":
            int(
                validation_date
                .isocalendar()
                .week
            ),

        # Promotions are available in the dataset
        # and can be treated as known future information.
        "onpromotion":
            validation_row["onpromotion"],

        # Future oil price is unknown.
        # Use latest known training-period value.
        "dcoilwtico":
            validation_oil,

        "is_holiday":
            validation_row["is_holiday"],

        "lag_1":
            lag_1,

        "lag_7":
            lag_7,

        "lag_14":
            lag_14,

        "lag_28":
            lag_28,

        "rolling_7":
            rolling_7,

        "rolling_14":
            rolling_14,

        "rolling_28":
            rolling_28

    }])


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(
        model_row[features]
    )[0]


    prediction = max(
        prediction,
        0
    )


    validation_predictions.append(
        prediction
    )


    # --------------------------------------------------------
    # RECURSIVE UPDATE
    # --------------------------------------------------------

    # IMPORTANT:
    # Add the prediction, not the actual sales.
    # This makes validation behave like the real
    # multi-step forecasting process.

    validation_history = pd.concat(
        [
            validation_history,

            pd.DataFrame({
                "date":
                    [validation_date],

                "sales":
                    [prediction]
            })
        ],
        ignore_index=True
    )


predictions = np.array(
    validation_predictions
)


# ============================================================
# 19. MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_valid,
    predictions
)


rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        predictions
    )
)


non_zero = (
    y_valid != 0
)


mape = (
    np.mean(
        np.abs(
            (
                y_valid[non_zero]
                - predictions[non_zero]
            )
            /
            y_valid[non_zero]
        )
    )
    * 100
)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    "MAE :",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "MAPE:",
    round(mape, 2),
    "%"
)


# ============================================================
# 20. ACTUAL VS PREDICTED VALIDATION
# ============================================================

validation_results = validation_data[
    ["date", "sales"]
].copy()


validation_results["predicted_sales"] = (
    predictions
)


plt.figure(figsize=(14, 6))


plt.plot(
    validation_results["date"],
    validation_results["sales"],
    label="Actual Sales"
)


plt.plot(
    validation_results["date"],
    validation_results["predicted_sales"],
    label="Predicted Sales"
)


plt.title(
    "Actual vs Predicted Daily Sales"
)

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "feature":
        features,

    "importance":
        model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print("\nFEATURE IMPORTANCE")

print(
    feature_importance
)


plt.figure(figsize=(10, 7))


plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)


plt.gca().invert_yaxis()


plt.title(
    "Forecast Model Feature Importance"
)

plt.xlabel(
    "Importance"
)


plt.tight_layout()

plt.show()


# ============================================================
# 22. PREPARE TEST DATA
# ============================================================

future = test[
    ["date"]
].drop_duplicates().copy()


future = future.sort_values(
    "date"
)


# ============================================================
# 23. FUTURE PROMOTIONS
# ============================================================

future_promotions = (
    daily_promotions_test
    .rename(
        columns={
            "onpromotion":
                "future_onpromotion"
        }
    )
)


future = future.merge(
    future_promotions,
    on="date",
    how="left"
)


future["future_onpromotion"] = (
    future["future_onpromotion"]
    .fillna(0)
)


# ============================================================
# 24. FUTURE HOLIDAY FLAG
# ============================================================

future["is_holiday"] = (
    future["date"]
    .isin(holiday_dates)
    .astype(int)
)


# ============================================================
# 25. FUTURE RECURSIVE FORECAST
# ============================================================

print("\nGenerating future forecasts...")

history = daily[
    ["date", "sales"]
].copy()


history = (
    history
    .sort_values("date")
    .reset_index(drop=True)
)


future_predictions = []


# Use latest known oil price because future oil prices
# are not available.

latest_oil = (
    daily["dcoilwtico"].iloc[-1]
)


for _, future_row in future.iterrows():

    next_date = (
        future_row["date"]
    )


    # --------------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------------

    lag_1 = (
        history["sales"].iloc[-1]
    )

    lag_7 = (
        history["sales"].iloc[-7]
    )

    lag_14 = (
        history["sales"].iloc[-14]
    )

    lag_28 = (
        history["sales"].iloc[-28]
    )


    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    rolling_7 = (
        history["sales"]
        .tail(7)
        .mean()
    )

    rolling_14 = (
        history["sales"]
        .tail(14)
        .mean()
    )

    rolling_28 = (
        history["sales"]
        .tail(28)
        .mean()
    )


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    model_row = pd.DataFrame([{

        "day_of_week":
            next_date.dayofweek,

        "day_of_month":
            next_date.day,

        "month":
            next_date.month,

        "quarter":
            next_date.quarter,

        "year":
            next_date.year,

        "week_of_year":
            int(
                next_date
                .isocalendar()
                .week
            ),

        "onpromotion":
            future_row[
                "future_onpromotion"
            ],

        "dcoilwtico":
            latest_oil,

        "is_holiday":
            future_row[
                "is_holiday"
            ],

        "lag_1":
            lag_1,

        "lag_7":
            lag_7,

        "lag_14":
            lag_14,

        "lag_28":
            lag_28,

        "rolling_7":
            rolling_7,

        "rolling_14":
            rolling_14,

        "rolling_28":
            rolling_28

    }])


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(
        model_row[features]
    )[0]


    prediction = max(
        prediction,
        0
    )


    future_predictions.append({

        "date":
            next_date,

        "forecast_sales":
            prediction

    })


    # --------------------------------------------------------
    # ADD PREDICTION TO HISTORY
    # --------------------------------------------------------

    history = pd.concat(
        [
            history,

            pd.DataFrame({

                "date":
                    [next_date],

                "sales":
                    [prediction]

            })
        ],
        ignore_index=True
    )


forecast = pd.DataFrame(
    future_predictions
)


# ============================================================
# 26. WHAT-IF SCENARIOS
# ============================================================

forecast["base_case"] = (
    forecast["forecast_sales"]
)


forecast["optimistic_10pct"] = (
    forecast["forecast_sales"]
    * 1.10
)


forecast["pessimistic_10pct"] = (
    forecast["forecast_sales"]
    * 0.90
)


forecast["optimistic_15pct"] = (
    forecast["forecast_sales"]
    * 1.15
)


forecast["pessimistic_15pct"] = (
    forecast["forecast_sales"]
    * 0.85
)


# ============================================================
# 27. FORECAST CHART
# ============================================================

plt.figure(figsize=(14, 7))


recent_history = (
    daily.tail(120)
)


plt.plot(
    recent_history["date"],
    recent_history["sales"],
    label="Historical Sales"
)


plt.plot(
    forecast["date"],
    forecast["base_case"],
    label="Base Forecast"
)


plt.plot(
    forecast["date"],
    forecast["optimistic_10pct"],
    label="Optimistic +10%"
)


plt.plot(
    forecast["date"],
    forecast["pessimistic_10pct"],
    label="Pessimistic -10%"
)


plt.title(
    "Store Sales Forecast & What-If Scenarios"
)

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 28. FORECAST SUMMARY
# ============================================================

base_total = (
    forecast["base_case"].sum()
)


optimistic_total = (
    forecast["optimistic_10pct"].sum()
)


pessimistic_total = (
    forecast["pessimistic_10pct"].sum()
)


print("\n================================")
print("FORECAST SUMMARY")
print("================================")


print(
    "Base Forecast:",
    round(
        base_total,
        2
    )
)


print(
    "Optimistic +10%:",
    round(
        optimistic_total,
        2
    )
)


print(
    "Pessimistic -10%:",
    round(
        pessimistic_total,
        2
    )
)


print(
    "\nPotential upside:",
    round(
        optimistic_total
        - base_total,
        2
    )
)


print(
    "Potential downside:",
    round(
        base_total
        - pessimistic_total,
        2
    )
)


# ============================================================
# 29. SAVE FORECAST RESULTS
# ============================================================

forecast.to_csv(
    "sales_forecast.csv",
    index=False
)


validation_results.to_csv(
    "forecast_validation.csv",
    index=False
)


feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)


daily.to_csv(
    "daily_sales_analysis.csv",
    index=False
)


# ============================================================
# 30. SAVE MODEL METRICS
# ============================================================

metrics = pd.DataFrame({

    "metric": [
        "MAE",
        "RMSE",
        "MAPE"
    ],

    "value": [
        mae,
        rmse,
        mape
    ]

})


metrics.to_csv(
    "forecast_metrics.csv",
    index=False
)


# ============================================================
# 31. FINISHED
# ============================================================

print("\n================================")
print("PROJECT PIPELINE COMPLETE")
print("================================")


print("""
Created files:

sales_forecast.csv
forecast_validation.csv
feature_importance.csv
daily_sales_analysis.csv
forecast_metrics.csv

Forecasting model:
Random Forest Regression

Validation:
60-day time-based recursive validation

Forecasting approach:
Recursive multi-step forecasting

What-If scenarios:
+10%
-10%
+15%
-15%

Next phase:
Streamlit What-If Forecasting Application
""")


print("\nAll forecasting processes completed successfully.")