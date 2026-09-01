# Retail Demand Forecasting System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-demand-forecaster.streamlit.app)
[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://retail-demand-forecaster.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning and Data Science pipeline that predicts daily store-level sales with **12.6% RMSPE** (a **31.6% improvement** over baseline) — simulating the core demand forecasting engines powering inventory operations at quick-commerce leaders like Blinkit, Swiggy Instamart, Zepto, and retail chains.

---

## 🔗 Live Interactive Demo

Experience the forecasting engine live in your browser:

👉 **[Launch Streamlit Dashboard (retail-demand-forecaster.streamlit.app)](https://retail-demand-forecaster.streamlit.app)** 🚀

> **Direct Link:** [https://retail-demand-forecaster.streamlit.app](https://retail-demand-forecaster.streamlit.app)

---

## 📖 About The Project

In the fast-moving retail and quick-commerce space, demand fluctuates dynamically across stores due to promotions, day-of-week trends, seasonal holidays, and local competitor density. Inaccurate predictions directly hit the bottom line through wasted inventory or lost revenue from stockouts.

**Retail Demand Forecasting System** solves this by providing:
- **Daily Store-Level Forecasts**: Granular predictions across 1,115 stores for 7, 14, or 30-day horizons.
- **High-Performance ML**: Gradient boosted decision trees (LightGBM) optimized with time-series validation.
- **Explainable AI (XAI)**: SHAP-based feature importance making every prediction transparent and auditable.
- **Interactive Decision Support**: A Streamlit dashboard allowing supply chain managers to simulate promotion impact in real-time.

---

## 📌 Problem Statement

Retail chains and dark stores face daily inventory trade-offs:

- ❌ **Overstocking**: Excess perishable stock expires, locking up working capital and inflating warehousing costs.
- ❌ **Understocking**: Stockouts lead to unfulfilled orders, lost revenue, and poor customer retention.

**Solution**: By harnessing historical transaction patterns, promo schedules, calendar cycles, and store metadata, this predictive system forecasts sales down to the individual store level with reliable confidence intervals.

---

## 📊 Dataset

Trained and evaluated on real historical retail transactions from the [Rossmann Store Sales Dataset](https://www.kaggle.com/c/rossmann-store-sales) on Kaggle.

| Metric | Value |
| :--- | :--- |
| **Total Rows (Cleaned)** | 844,338 |
| **Number of Unique Stores** | 1,115 |
| **Time Period** | January 2013 – July 2015 |
| **Original Features** | 9 |
| **Engineered Features** | 27 |
| **Target Variable** | Daily Store Sales (€) |

---

## 🏗️ Project Architecture

```
Raw Data (train.csv, store.csv)
       │
       ▼
Data Cleaning & Preprocessing (handling missing competition data, date parsing)
       │
       ▼
Exploratory Data Analysis (distribution transforms, seasonality checks, promo lifts)
       │
       ▼
Feature Engineering (27 lag, rolling, calendar, and interaction features)
       │
       ▼
Model Training & Validation (Strict time-series split: last 6 weeks held out)
 ├── Baseline: Linear Regression
 ├── Time-Series Decomposition: Facebook Prophet
 └── Primary Model: LightGBM Regressor (995 sequential trees)
       │
       ▼
Model Explainability (SHAP summary & waterfall breakdowns)
       │
       ▼
Interactive Web Application (Streamlit Cloud Dashboard with Real-Time Scenario Simulation)
```

---

## 🔍 Exploratory Data Analysis (EDA)

Key business and statistical insights uncovered during exploratory analysis:

| Insight | Key Finding |
| :--- | :--- |
| **Promotion Lift** | **+38.8% higher sales** on active promotional days compared to non-promo days. |
| **Busiest Day** | **Monday** exhibits the highest sales volume across all store categories. |
| **Peak Season** | **December (+45%)** — dramatic holiday spike requiring preemptive stock build-up. |
| **Store Type Dynamics** | **Type B** stores average **€10,233/day**, significantly outpacing others (€6,900/day). |
| **Competition Impact** | Stores with competitors further than 5 km consistently register higher daily volumes. |
| **Target Distribution** | Daily sales exhibit right-skew; log-transformation normalizes variance for regression. |

---

## ⚙️ Feature Engineering

From 9 raw attributes, **27 high-signal features** were engineered to capture temporal dynamics, seasonal cycles, and cross-feature interactions:

| Category | Features | Purpose & Intuition |
| :--- | :--- | :--- |
| **Lag Features** | `Sales_Lag1`, `Lag7`, `Lag14`, `Lag30` | Capture sales momentum, short-term trends, and strong autocorrelation. |
| **Rolling Window** | `Rolling_Mean_7`, `Mean_14`, `Std_7`, `Max_7` | Smooth out high-frequency noise and represent recent sales velocity. |
| **Date & Calendar** | `Year`, `Month`, `Day`, `WeekOfYear`, `DayOfYear`, `IsWeekend`, `IsMonthStart`, `IsMonthEnd` | Account for weekly, monthly, and annual seasonal demand shifts. |
| **Domain Interactions**| `Promo_StoreType`, `Promo_DayOfWeek` | Capture how promotions have varying effectiveness across store types and days. |
| **Categorical Encoding**| `StoreType_enc`, `Assortment_enc`, `StateHoliday_enc` | Numerically encode store assortment and holiday classifications. |

> [!IMPORTANT]
> **No Data Leakage**: A strict time-based split was implemented (NOT a random train/test split). The final 6 weeks of data were held out as the validation set to mirror true production forecasting conditions.

---

## 🤖 Modeling & Performance

Three model architectures were tested and benchmarked:

### 1. Linear Regression (Baseline)
- Serves as the minimum benchmark.
- **RMSE**: €1,245 | **RMSPE**: 20.6%

### 2. LightGBM (Primary Model)
- High-efficiency gradient boosting with 995 sequential decision trees.
- Natively captures non-linear relationships, multi-level feature interactions, and tabular nuances.
- Tuned with early stopping on validation loss to prevent overfitting.
- **RMSE**: €851 | **RMSPE**: 12.6%
- **Result**: **31.6% performance improvement** over baseline.

```python
# Primary Model Configuration
LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
```

### 3. Facebook Prophet (Decomposition & Bounds)
- Decomposes sales time series into additive trend, weekly cycles, and yearly seasonality.
- Confirms the Monday weekly peak and December (+45%) holiday surge.
- Generates historical confidence bands for uncertainty estimation.

---

## 📈 Results Comparison

| Model | RMSE (€) | RMSPE (%) | Improvement |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | €1,245 | 20.6% | — |
| **LightGBM (Primary Model)** | **€851** | **12.6%** | **+31.6%** |

- **Training Samples**: 770,117
- **Validation Samples**: 58,611 (last 6 calendar weeks)
- **Features Used**: 27 engineered features

---

## 🧠 Explainable AI (SHAP)

To make forecasts actionable for non-technical retail operators, models are paired with **SHAP (SHapley Additive exPlanations)**:

| Rank | Feature | Importance | Actionable Business Takeaway |
| :---: | :--- | :---: | :--- |
| **1** | `Rolling_Mean_14` | **Highest** | The trailing 2-week baseline sales level is the single strongest anchor. |
| **2** | `Promo` | **Very High** | Quantifies the massive demand surge on active campaign days. |
| **3** | `Sales_Lag1` | **High** | Immediate prior-day sales provide essential short-term inertia. |
| **4** | `DayOfWeek` | **High** | Strong intra-week cyclic patterns (Monday spike vs weekend dips). |
| **5** | `Promo_DayOfWeek` | **Medium** | Custom interaction feature confirms promos are more potent on specific days. |

### Sample Waterfall Breakdown:
For an illustrative daily store prediction of **€6,137** (Actual: €5,201):
- `Promo = 0` contributed **-€436** (absence of promotion lowered expected demand).
- `Sales_Lag1 = 5,591` contributed **-€376** (lower previous day pulled prediction down).
- `Rolling_Mean_14 = 7,296` contributed **+€347** (strong two-week average anchored upwards).

---

## 🖥️ Streamlit Dashboard Features

The deployed web application provides supply chain planners with interactive tooling:

- 🏬 **Store Selector**: Choose from all 1,115 individual retail stores.
- ⏱️ **Configurable Forecast Horizon**: Forecast 7, 14, or 30 days ahead into the future.
- 🎯 **What-If Promo Simulation**: Toggle promotions ON/OFF in real time to simulate demand elasticity before locking in marketing spend.
- 📊 **Confidence Intervals**: Visual upper and lower prediction bounds for risk-aware inventory buffering.
- 📋 **Exportable Forecast Tables**: Exact date-by-date predicted euro values and error bounds.
- 💡 **Store Analytics**: Day-of-week breakdown and historical monthly sales trajectories.

👉 **Try it here**: [retail-demand-forecaster.streamlit.app](https://retail-demand-forecaster.streamlit.app)

---

## 🛠️ Tech Stack

| Category | Technologies / Libraries |
| :--- | :--- |
| **Language** | Python 3.11 |
| **Machine Learning** | LightGBM, Scikit-learn |
| **Time Series** | Facebook Prophet, Statsmodels |
| **Model Interpretability**| SHAP (SHapley Additive exPlanations) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly Express |
| **Web Dashboard** | Streamlit |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git, GitHub |

---

## 📁 Repository Structure

```
retail-demand-forecaster/
│
├── app.py                      # Interactive Streamlit web application
├── requirements.txt            # Python dependencies and locked versions
├── README.md                   # Project documentation and architecture
│
└── models/
    ├── lgbm_model.pkl          # Pre-trained LightGBM regression model
    ├── features.pkl            # Pickled list of 27 feature column names
    ├── metrics.pkl             # Model performance metrics dictionary
    └── train_dashboard.csv     # Sample historical dataset for dashboard caching
```

---

## 🚀 Getting Started (Run Locally)

Clone the repository and launch the dashboard locally in three simple steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Number789Alpha/retail-demand-forecaster.git
cd retail-demand-forecaster
```

### 2. Install Dependencies
Ensure you have Python 3.10+ or 3.11 installed, then install required packages:
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will start locally at `http://localhost:8501`.

---

## 📓 Training Pipeline & Colab Notebook

The complete end-to-end model development lifecycle (ETL, exploratory data analysis, feature engineering, cross-validation, and SHAP computation) was authored in Google Colab:

- **Sections 1–5**: Environment setup, dependency loading, and CSV ingestion
- **Sections 6–10**: Handling missing values, outlier detection, and data merging
- **Sections 11–20**: Comprehensive EDA, distribution transformation, and visual correlation
- **Sections 21–25**: Time-series feature engineering (lags, rolling stats, interactions)
- **Sections 26–30**: Baseline vs LightGBM training with early stopping
- **Sections 31–35**: SHAP tree explanation and Prophet seasonality modeling
- **Sections 36–38**: Model serialization (`pkl` exports) for dashboard deployment

---

## 💡 Key Engineering Takeaways

1. **Feature Engineering > Algorithm Complexity**: The 31.6% performance leap was primarily driven by domain-aware rolling windows and lag features rather than hyperparameter tuning alone.
2. **Preventing Temporal Leakage**: Standard k-fold cross-validation is fatally flawed for time series; chronological splits are critical to prevent future data from contaminating training.
3. **Domain Interaction Features**: Custom features like `Promo_DayOfWeek` ranked in the top 5 SHAP values, highlighting that promotional lift is day-dependent.
4. **Explainability as a First-Class Citizen**: Black-box models struggle with operational adoption; SHAP bridges the gap between data science and warehouse managers.

---

## 🎯 Business Value

| Operational Metric | Practical Benefit |
| :--- | :--- |
| **12.6% Average Error** | Reliable enough for daily dark store ordering schedules. |
| **Interactive Promo Toggle** | Estimates promo ROI and volume lifts before promotional budgets are committed. |
| **Store-Level Precision** | 1,115 independent store models rather than generic regional aggregations. |
| **Interpretable Predictions** | Every prediction displays positive and negative contributing factors. |

---

## 👨‍💻 Author

**Priyank Mishra**  
- GitHub: [@Number789Alpha](https://github.com/Number789Alpha)  
- Live App: [retail-demand-forecaster.streamlit.app](https://retail-demand-forecaster.streamlit.app)
