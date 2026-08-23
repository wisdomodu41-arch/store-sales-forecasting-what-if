# 📈 Store Sales Forecasting & What-If Analytics

## Executive Summary

This project develops an end-to-end **machine learning sales forecasting and business scenario analysis solution** for retail store sales.

The solution combines historical sales data with promotions, transactions, oil prices, holidays, calendar-based features, lag variables, and rolling statistics to forecast future sales using a **Random Forest Regression model**.

The project goes beyond forecasting by providing an interactive **Streamlit What-If Analytics application**, allowing users to adjust expected sales growth or decline and immediately evaluate the potential business impact.

The objective is to transform historical retail data into actionable insights that can support:

- Sales and revenue planning
- Inventory and demand planning
- Promotional decision-making
- Operational resource allocation
- Scenario analysis
- Forecast-driven business planning

---

## 🎯 Business Problem

Retail businesses need reliable estimates of future demand to make better operational and commercial decisions.

However, sales can be influenced by multiple factors, including:

- Historical sales patterns
- Promotions
- Customer transactions
- Holidays
- Seasonal behavior
- Calendar effects
- External economic indicators such as oil prices

A simple historical average may not adequately capture these relationships.

This project addresses the problem by building a machine-learning forecasting pipeline capable of learning relationships between historical sales patterns and explanatory variables, while also providing a business-friendly interface for scenario analysis.

---

## 💡 Solution

The project consists of two major components:

### 1. Machine Learning Forecasting Pipeline

A Python forecasting pipeline was developed to:

1. Load and inspect multiple datasets
2. Clean and transform the data
3. Aggregate store-level sales into daily sales
4. Integrate promotions, transactions, oil prices, and holidays
5. Engineer calendar-based features
6. Create lag and rolling-window features
7. Perform a time-based validation split
8. Train a Random Forest regression model
9. Evaluate forecasting performance
10. Generate future sales forecasts
11. Identify the most influential forecasting features
12. Produce What-If scenarios

### 2. Interactive Streamlit Application

The forecasting outputs are integrated into an interactive Streamlit application that allows users to:

- Select the forecast horizon
- Adjust a What-If growth or decline assumption
- Compare base forecasts against scenarios
- View projected sales impact
- Examine model validation results
- Review model performance metrics
- Explore important forecasting drivers

---

# 🧠 Analytical Approach

## 1. Data Preparation

The forecasting pipeline combines multiple datasets covering:

- Historical sales
- Store information
- Promotions
- Transactions
- Oil prices
- Holidays and events

Date fields were converted into datetime format to enable temporal analysis and feature engineering.

Missing oil price observations were handled using forward and backward filling.

---

## 2. Daily Sales Aggregation

Store-level sales were aggregated by date to create a daily sales series.

Daily promotional activity and transactions were also aggregated to create explanatory variables aligned with the sales timeline.

---

## 3. Feature Engineering

Several categories of predictive features were created.

### Calendar Features

- Day of week
- Day of month
- Month
- Quarter
- Year
- Week of year
- Holiday indicator

### Lag Features

Historical sales were shifted to capture previous demand patterns:

- 1-day lag
- 7-day lag
- 14-day lag
- 28-day lag

### Rolling Features

Historical moving averages were created using:

- 7-day rolling average
- 14-day rolling average
- 28-day rolling average

The rolling calculations use previous observations to avoid directly incorporating the current target value.

### External Variables

The model also incorporates:

- Promotions
- Transactions
- Oil prices
- Holiday indicators

---

# 🤖 Machine Learning Model

## Random Forest Regression

A **Random Forest Regressor** was selected as the forecasting model.

The model was configured with:

- 200 trees
- Maximum tree depth of 18
- Minimum samples per leaf of 2
- Fixed random state for reproducibility
- Parallel processing enabled

Random Forest was selected because it can model nonlinear relationships and interactions between multiple forecasting variables without requiring strict linear assumptions.

---

# 📊 Model Validation

Because this is a time-series forecasting problem, the data was **not randomly shuffled**.

Instead, a chronological validation strategy was used.

The final **60 days of the available modeling period** were held out for validation, while earlier observations were used for model training.

This approach better reflects the real-world forecasting situation where historical information is used to predict future observations.

### Evaluation Metrics

The model is evaluated using:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| RMSE | Penalizes larger prediction errors more heavily |
| MAPE | Measures average percentage error |

The resulting metrics are available in:

`forecast_metrics.csv`

---

# 🔍 Feature Importance

The model's feature importance scores are calculated to identify which variables contribute most strongly to the Random Forest's predictive decisions.

This provides an additional layer of business interpretation by showing which historical patterns and explanatory variables are most influential in the forecasting model.

Results are available in:

`feature_importance.csv`

---

# 🔮 Future Forecasting

After model validation, the pipeline generates forecasts for the future test period.

Future predictions are generated recursively:

1. The model predicts the next date.
2. That prediction is added to the historical sequence.
3. Updated lag and rolling features are calculated.
4. The process continues through the forecast horizon.

Where future transaction and oil-price information is unavailable, recent historical values are used as assumptions.

This allows the model to produce a complete future forecast while making the assumptions explicit.

---

# 📈 What-If Scenario Analysis

The project extends traditional forecasting with scenario analysis.

The base forecast can be adjusted using a user-selected growth or decline assumption.

For example:

- **+10%** → optimistic scenario
- **0%** → base forecast
- **-10%** → pessimistic scenario
- Custom values → user-defined scenario

The application calculates:

- Base projected sales
- Scenario projected sales
- Absolute scenario impact
- Percentage change

This allows business users to answer questions such as:

> "What happens to projected sales if demand increases by 10%?"

or:

> "What would the expected sales impact be if demand falls by 15%?"

---

# 🖥️ Streamlit Analytics Application

The interactive application provides a business-facing interface for the forecasting model.

### Main Components

#### Forecast Controls

Users can adjust:

- Forecast horizon
- What-If growth rate

#### Forecast Overview

The application displays:

- Base Forecast
- Scenario Forecast
- Scenario Impact
- Forecast Period

#### Forecast Trend

Users can visually compare:

- Base forecast
- Selected What-If scenario

#### Scenario Analysis

The application compares:

- Pessimistic scenario
- Base forecast
- Optimistic scenario
- Selected user scenario

#### Model Validation

Actual sales are compared with predicted sales to visually assess model performance.

#### Model Performance

The application displays:

- MAE
- RMSE
- MAPE

#### Forecasting Drivers

The application visualizes the most important model features.

---

# 📁 Project Structure

```text
store-sales-forecasting-what-if/
│
├── app.py
├── forecasting_project.py
├── requirements.txt
├── README.md
│
├── sales_forecast.csv
├── forecast_validation.csv
├── forecast_metrics.csv
├── feature_importance.csv
└── daily_sales_analysis.csv
📄 Output Files
sales_forecast.csv
Contains future sales forecasts and What-If scenario projections.
forecast_validation.csv
Contains actual versus predicted sales for the validation period.
forecast_metrics.csv
Contains the model evaluation metrics:
MAE
RMSE
MAPE
feature_importance.csv
Contains the Random Forest feature importance results.
daily_sales_analysis.csv
Contains the engineered daily sales dataset used throughout the forecasting analysis.
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Streamlit
Machine Learning
Time-Series Feature Engineering
Data Analysis
Scenario Analysis
▶️ How to Run the Project
1. Clone the Repository
git clone https://github.com/YOUR-USERNAME/store-sales-forecasting-what-if.git
Navigate into the project directory:
cd store-sales-forecasting-what-if
2. Install Dependencies
pip install -r requirements.txt
3. Run the Streamlit Application
streamlit run app.py
The application will open in your browser.
📌 Key Analytical Considerations
Time-Series Validation
Random train/test splitting was avoided because it can introduce future information into the training process.
A chronological validation strategy was therefore used.
Recursive Forecasting
Future predictions are generated sequentially, with previous predictions becoming part of the history used to generate subsequent forecasts.
Future Data Assumptions
Some future explanatory variables are unavailable at forecasting time.
For this reason:
Recent transaction averages are used for future transactions.
The latest available oil price is used as the future oil-price assumption.
These assumptions should be replaced with actual future estimates when deploying the solution in a production environment.
💼 Business Value
This project demonstrates how machine learning can be translated into practical business decision support.
Potential business applications include:
Demand Planning
Estimate expected future sales to support inventory planning.
Promotion Planning
Evaluate potential sales changes under different demand scenarios.
Resource Planning
Use forecasted demand to support staffing and operational planning.
Risk Assessment
Understand potential downside scenarios when expected demand decreases.
Strategic Planning
Compare baseline expectations with optimistic and pessimistic scenarios before making business decisions.
🚀 Potential Future Improvements
The current solution provides a strong analytical foundation, but several improvements could make it more production-ready.
Future development could include:
Store-level forecasting
Product-family forecasting
Advanced time-series models
XGBoost or LightGBM comparison
Hyperparameter optimization
Cross-validation designed for time series
Prediction intervals and uncertainty estimates
Automated model retraining
More sophisticated holiday effects
Future transaction forecasting
Automated external-variable forecasting
Forecast monitoring and drift detection
Deployment to a cloud platform
Automated data pipelines
🎓 Skills Demonstrated
This project demonstrates practical experience in:
Data cleaning
Exploratory data preparation
Feature engineering
Time-series analysis
Machine learning
Regression modeling
Model validation
Forecast evaluation
Recursive forecasting
Feature importance analysis
Scenario analysis
Business analytics
Data visualization
Python
Streamlit
Analytical storytelling
👤 Author
Wisdom Odu
Aspiring Data Analyst focused on:
Data Analytics • Business Intelligence • Machine Learning • Forecasting • Decision Analytics
GitHub:
https://github.com/wisdomodu41-arch⁠�
⭐ Project Overview
Store Sales Forecasting & What-If Analytics transforms historical retail data into a machine-learning forecasting solution and interactive decision-support application.
The project demonstrates an end-to-end workflow from data preparation → feature engineering → model development → time-based validation → forecasting → scenario analysis → business-facing Streamlit application.