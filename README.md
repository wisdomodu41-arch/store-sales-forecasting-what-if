# 📈 Store Sales Forecasting & What-If Analytics

## Executive Summary

This project is an end-to-end **retail sales forecasting and decision-support solution** designed to help businesses anticipate future sales, evaluate forecasting reliability, and quantify the potential impact of different business scenarios.

The project combines **Python, machine learning, time-series feature engineering, model validation, and Streamlit** to transform historical retail sales data into actionable forecasts.

Beyond simply predicting future sales, the solution provides an interactive **What-If analysis layer**, allowing decision-makers to test potential changes in sales growth and immediately see their projected financial impact.

The result is a practical analytics application that connects **predictive modelling with business decision-making**.

---

## 🎯 Business Problem

Retail businesses need accurate sales forecasts to support decisions involving:

- Inventory planning
- Staffing and workforce allocation
- Promotion planning
- Revenue forecasting
- Operational capacity
- Supply planning
- Budgeting
- Risk management

Relying only on historical averages can fail to capture important patterns such as:

- Day-of-week effects
- Seasonal behaviour
- Promotional activity
- Recent sales momentum
- Transaction activity
- External economic conditions
- Holiday periods

This project addresses the problem by developing a machine-learning forecasting pipeline that incorporates historical sales patterns and relevant business drivers.

---

## 💡 Project Objective

The primary objectives were to:

1. Build an end-to-end retail sales forecasting pipeline.
2. Engineer meaningful time-series and business features.
3. Train a machine-learning forecasting model.
4. Evaluate the model using time-based validation.
5. Generate future sales forecasts.
6. Identify the most influential forecasting drivers.
7. Develop an interactive What-If analysis application.
8. Translate forecast results into business-oriented insights.

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Data processing, modelling and analytics |
| **Pandas** | Data manipulation and feature engineering |
| **NumPy** | Numerical computation |
| **Scikit-learn** | Machine-learning model development |
| **Random Forest Regressor** | Sales forecasting |
| **Matplotlib** | Forecast and model visualisation |
| **Streamlit** | Interactive analytics application |
| **CSV** | Data and analytical outputs |
| **GitHub** | Version control and project portfolio |

---

# 🔄 Analytical Workflow

The project follows an end-to-end analytical workflow:

```text
Raw Retail Data
       ↓
Data Preparation
       ↓
Data Integration
       ↓
Feature Engineering
       ↓
Time-Based Validation
       ↓
Random Forest Forecasting Model
       ↓
Model Evaluation
       ↓
Future Sales Forecast
       ↓
Feature Importance Analysis
       ↓
What-If Scenario Analysis
       ↓
Interactive Streamlit Application
       ↓
Business Decision Support
📊 Data Preparation
Multiple datasets were integrated to create a unified forecasting dataset.
The analysis incorporates information relating to:
Historical sales
Store activity
Promotions
Transactions
Oil prices
Holiday events
Dates were standardized and the datasets were aggregated to the daily level for forecasting analysis.
Missing explanatory values were handled using appropriate techniques, including forward/backward filling for oil-price data and zero-filling where the absence of promotional or transaction activity represented no recorded activity.
🧠 Feature Engineering
Several forecasting features were created to capture temporal patterns and recent sales behaviour.
Calendar Features
Day of week
Day of month
Month
Quarter
Year
Week of year
Holiday indicator
Sales Lag Features
Historical sales values were incorporated through:
1-day lag
7-day lag
14-day lag
28-day lag
These features allow the model to capture recent and recurring sales patterns.
Rolling Features
Rolling averages were also created:
7-day rolling average
14-day rolling average
28-day rolling average
These features help represent short-, medium-, and longer-term sales momentum.
Business Drivers
Additional explanatory variables include:
Promotion activity
Transaction volume
Oil price
🤖 Forecasting Model
A Random Forest Regression model was selected for the forecasting task.
The model was configured with:
200 decision trees
Maximum tree depth of 18
Minimum samples per leaf of 2
Fixed random state for reproducibility
Parallel processing
Random Forest was selected because it can model complex non-linear relationships between sales and multiple business drivers while providing feature-importance information that can support business interpretation.
🧪 Model Validation
Because this is a time-series forecasting problem, the project avoids random train/test shuffling.
Instead, a time-based validation strategy was used.
The most recent 60 days of available historical data were separated as the validation period, while earlier observations were used for model training.
This approach better reflects the real-world forecasting situation:
Train on the past → predict a future period → compare predictions with actual outcomes.
Evaluation Metrics
The model is evaluated using:
MAE — Mean Absolute Error
Measures the average absolute difference between actual and predicted sales.
RMSE — Root Mean Squared Error
Penalizes larger prediction errors more heavily and helps identify whether the model occasionally makes substantial forecasting mistakes.
MAPE — Mean Absolute Percentage Error
Expresses prediction error as a percentage, making performance easier to interpret from a business perspective.
The calculated metrics are available in:
forecast_metrics.csv
📈 Forecast Results
The forecasting pipeline generates future sales predictions using a recursive forecasting approach.
The forecast output contains:
Forecast date
Base forecast
Optimistic scenarios
Pessimistic scenarios
The generated forecast is stored in:
/sales_forecast.csv
The validation results are stored in:
/forecast_validation.csv
🔎 Forecasting Drivers
Feature importance analysis is used to identify which variables contributed most strongly to the Random Forest model's predictions.
This provides an additional layer of business insight beyond the forecast itself.
The analysis helps answer questions such as:
Which historical sales patterns are most influential?
How important is recent sales activity?
Does promotional activity contribute strongly to forecasting?
Which time-related variables influence predictions?
The results are available in:
feature_importance.csv
🎛️ What-If Scenario Analysis
Forecasting alone tells a business what may happen.
What-If analysis helps explore:
"What could happen if conditions change?"
The Streamlit application allows users to adjust a configurable growth rate and compare the resulting scenario against the base forecast.
For example:
Base Scenario
Expected sales based on the forecasting model.
Upside Scenario
Sales assuming positive growth relative to the base forecast.
Downside Scenario
Sales assuming negative growth relative to the base forecast.
The application calculates:
Base projected sales
Scenario projected sales
Absolute scenario impact
Percentage change
Forecast horizon
This allows users to evaluate potential upside and downside outcomes without retraining the forecasting model.
🖥️ Interactive Streamlit Application
The project includes an interactive Streamlit application designed as a business-facing forecasting dashboard.
The application provides:
Forecast Controls
Users can adjust:
Forecast horizon
What-If growth rate
Forecast Overview
Key performance indicators display:
Base forecast
Scenario forecast
Scenario impact
Forecast period
Forecast Trend
Interactive visual comparison of:
Base forecast
Selected What-If scenario
Scenario Analysis
Comparison of:
Pessimistic scenario
Base forecast
Optimistic scenario
Selected What-If scenario
Model Validation
The application displays:
Actual sales
Predicted sales
Model performance metrics
Forecasting Drivers
The application visualizes the most important features used by the model.
📁 Project Structure
store-sales-forecasting-what-if/
│
├── README.md
├── app.py
├── forecasting_project.py
├── requirements.txt
│
├── sales_forecast.csv
├── forecast_validation.csv
├── forecast_metrics.csv
├── feature_importance.csv
└── daily_sales_analysis.csv
Core Files
forecasting_project.py
Main forecasting pipeline containing:
Data preparation
Feature engineering
Model training
Validation
Forecast generation
Feature importance
What-If scenario calculations
Output generation
app.py
Streamlit application providing the interactive forecasting and What-If analytics interface.
requirements.txt
Python dependencies required to run the project.
Analytical Outputs
sales_forecast.csv
Future sales forecast and scenario outputs.
forecast_validation.csv
Actual versus predicted sales during the validation period.
forecast_metrics.csv
MAE, RMSE and MAPE model-performance metrics.
feature_importance.csv
Model-derived forecasting feature importance.
daily_sales_analysis.csv
Prepared daily analytical dataset containing sales, business drivers and engineered features.
Raw source datasets are not included in this repository. The repository focuses on the analytical pipeline, model outputs and interactive application.
▶️ How to Run the Project
1. Clone the repository
git clone <repository-url>
2. Navigate to the project directory
cd store-sales-forecasting-what-if
3. Install the required dependencies
pip install -r requirements.txt
4. Run the forecasting pipeline
python forecasting_project.py
This generates the analytical output CSV files.
5. Launch the Streamlit application
streamlit run app.py
The application will open in your browser.
📌 Key Business Applications
The solution can support several real-world retail decisions.
Inventory Planning
Forecasted demand can help businesses prepare inventory for expected sales levels.
Promotion Planning
Businesses can evaluate how changes in sales assumptions could affect projected revenue.
Workforce Planning
Expected demand can support staffing and operational capacity decisions.
Revenue Planning
Scenario analysis allows management to estimate potential upside and downside outcomes.
Risk Management
Downside scenarios can help identify potential exposure if expected sales growth does not materialize.
📈 Business Value
The main value of this project is not simply producing a sales prediction.
It combines:
Descriptive Analytics
Understanding historical sales behaviour.
↓
Predictive Analytics
Estimating future sales.
↓
Diagnostic Analysis
Understanding the drivers behind model predictions.
↓
Prescriptive / Scenario Analysis
Evaluating potential outcomes under different assumptions.
↓
Decision Support
Providing an interactive interface that allows business users to explore potential outcomes.
This transforms a forecasting model into a more practical business decision-support tool.
🎯 Skills Demonstrated
This project demonstrates practical experience in:
Python data analysis
Pandas
NumPy
Data cleaning
Data integration
Feature engineering
Time-series analysis
Lag and rolling features
Machine learning
Random Forest regression
Model validation
MAE, RMSE and MAPE
Feature importance analysis
Forecasting
Scenario analysis
Business interpretation
Data visualization
Streamlit application development
Analytical storytelling
GitHub project documentation
🚀 Future Improvements
Potential future enhancements include:
Store-level forecasting
Category-level forecasting
Advanced time-series models
Automated model retraining
Prediction intervals
Probabilistic forecasting
Automated anomaly detection
Promotion-effect analysis
More advanced scenario drivers
Cloud deployment
Automated data pipelines
👤 Author
Wisdom Odu
Aspiring Data Analyst focused on transforming business data into actionable insights through:
Python • SQL • Power BI • Machine Learning • Forecasting • Business Analytics
⭐ Project Focus
Retail Sales Forecasting | Predictive Analytics | What-If Analysis | Decision Support
This project demonstrates an end-to-end approach to solving a real-world business problem — from data preparation and predictive modelling through to interactive analytics and business decision support.