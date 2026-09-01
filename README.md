# 🛒 Retail Demand Forecasting System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-demand-forecaster.streamlit.app)
[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://retail-demand-forecaster.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-LightGBM-brightgreen?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-orange?style=for-the-badge)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning and Data Science pipeline that predicts daily store-level sales with **12.6% RMSPE** (a **31.6% improvement** over baseline) — simulating the core demand forecasting engines powering inventory operations at quick-commerce leaders like Blinkit, Swiggy Instamart, Zepto, and retail chains.

---

## 🔗 Live Interactive Demo

Experience the forecasting engine live in your browser:

👉 **[Launch Streamlit Dashboard (retail-demand-forecaster.streamlit.app)](https://retail-demand-forecaster.streamlit.app)** 🚀

> **Direct URL:** [https://retail-demand-forecaster.streamlit.app](https://retail-demand-forecaster.streamlit.app)

---

## 🎯 What Exactly Is This Project?

At its core, this project is a **production-ready, hyper-local demand forecasting and decision-support system** built to solve the inventory dilemma in retail and quick-commerce.

```
+----------------------------------------------------------------------------------------------------+
|                                    RETAIL DEMAND FORECASTER                                        |
|                                                                                                    |
|  [ Historical Sales ] ──> [ 27 Engineered Features ] ──> [ LightGBM Regressor ] ──> [ Forecasts ]  |
|  (844K rows, 1,115 stores) (Lags, Rolling, Interactions)    (12.6% RMSPE, 995 trees) (7/14/30 Days)  |
|                                                                     │                              |
|                                                                     ▼                              |
|                                                         [ Interactive Streamlit App ]               |
|                                                         • What-If Promo Simulation                  |
|                                                         • ±12% Confidence Bands                     |
|                                                         • Day-of-Week & Seasonal Analytics          |
+----------------------------------------------------------------------------------------------------+
```

### In Plain Terms:
1. **The Problem**: A retail chain with 1,115 stores needs to know exactly how many euros of inventory each store will sell every day next week or next month. If they order too much, perishables rot and capital is locked up. If they order too little, shelves go empty and customers switch to competitors.
2. **What the Machine Learning Does**: It analyzes 844,338 historical sales records, learns how day-of-the-week, nearby competitors, holidays, promotions, and recent sales trends affect demand, and generates future sales predictions day-by-day.
3. **What the Interactive Web App Does**: Allows a supply chain manager to pick any store (1 to 1,115), choose a 7, 14, or 30-day forecast window, and toggle **"Promo Active"** on or off to instantly see how running a promotional campaign will spike demand and affect inventory requirements.

---

## 📖 About The Project

In high-velocity commerce (dark stores, supermarkets, pharmacies), demand fluctuates dynamically across stores due to promotions, local demographics, day-of-week trends, seasonal holidays, and competitor proximity.

**Retail Demand Forecasting System** bridges the gap between raw machine learning research and real-world supply chain decision-making:
- **Granular Store-Level Predictions**: Independent forecasting across 1,115 stores rather than crude aggregate company estimates.
- **Recursive Multi-Step Forecasting**: Automatically feeds prior predictions forward to update rolling averages (`Rolling_Mean_7`, `Rolling_Mean_14`) and lag terms (`Sales_Lag1` to `Sales_Lag30`).
- **High-Performance Gradient Boosting**: LightGBM regressor with 995 sequential decision trees, cutting baseline RMSE from €1,245 down to €851.
- **Explainable AI (XAI)**: SHAP-based feature attribution making model reasoning transparent, auditable, and trustworthy for non-technical stakeholders.
- **Dynamic What-If Simulation**: Instant preview of expected promotional lift (+38.8% average) to guide marketing spend and procurement buffers.

---

## 📌 Problem Statement

Retail chains and quick-commerce dark stores operate on razor-thin margins and face daily inventory penalties:

- ❌ **Overstocking**: Working capital tied down, warehouse capacity choked, and perishable goods marked down or discarded.
- ❌ **Understocking (Stockouts)**: Lost revenue, unfulfilled orders, and damaged customer loyalty in quick-delivery ecosystems.

**Solution**: By transforming historical sales data into 27 predictive features, this system produces daily sales forecasts paired with uncertainty bands (±12%) so operations teams can order stock with statistical confidence.

---

## 📊 Dataset & Characteristics

Trained and evaluated on real historical retail transactions from the [Rossmann Store Sales Dataset](https://www.kaggle.com/c/rossmann-store-sales) on Kaggle.

| Metric | Value |
| :--- | :--- |
| **Total Rows (After Cleaning)** | 844,338 |
| **Number of Unique Stores** | 1,115 |
| **Time Period** | January 2013 – July 2015 |
| **Original Features** | 9 (Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday) |
| **Engineered Features** | 27 (Lags, Rolling stats, Calendar components, Interactions) |
| **Target Variable** | Daily Sales (€) |

---

## 🏗️ Project Architecture & Pipeline

```
Raw Data (train.csv, store.csv)
       │
       ▼
Data Cleaning & Preprocessing (handling missing competition distance, date parsing, zero-sales filtering)
       │
       ▼
Exploratory Data Analysis (distribution transforms, seasonality checks, promo lifts)
       │
       ▼
Feature Engineering (27 lag, rolling, calendar, and interaction features)
       │
       ▼
Model Training & Validation (Strict time-based split: last 6 calendar weeks held out)
 ├── Baseline: Linear Regression (€1,245 RMSE)
 ├── Time-Series Decomposition: Facebook Prophet (Trend + Weekly + Yearly cycles)
 └── Primary Model: LightGBM Regressor (€851 RMSE, 12.6% RMSPE)
       │
       ▼
Model Explainability (SHAP TreeExplainer: summary plots & waterfall breakdowns)
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

## ⚙️ Feature Engineering (27 Features)

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

---

## 🖥️ Streamlit Dashboard Features

The deployed web application provides supply chain planners with interactive tooling:

- 🏬 **Store Selector**: Choose from all 1,115 individual retail stores.
- ⏱️ **Configurable Forecast Horizon**: Forecast 7, 14, or 30 days ahead into the future.
- 🎯 **What-If Promo Simulation**: Toggle promotions ON/OFF in real time to simulate demand elasticity before locking in marketing spend.
- 📊 **Confidence Intervals**: Visual upper and lower prediction bounds (±12%) for risk-aware inventory buffering.
- 📋 **Exportable Forecast Tables**: Exact date-by-date predicted euro values and error bounds.
- 💡 **Store Analytics**: Day-of-week breakdown and historical monthly sales trajectories.

👉 **Try it live**: [retail-demand-forecaster.streamlit.app](https://retail-demand-forecaster.streamlit.app)

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
    └── train_dashboard.csv     # Historical dataset cache for dashboard
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
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will start locally at `http://localhost:8501`.

---

## 💡 Key Engineering Takeaways

1. **Feature Engineering > Algorithm Complexity**: The 31.6% performance leap was primarily driven by domain-aware rolling windows and lag features rather than hyperparameter tuning alone.
2. **Preventing Temporal Leakage**: Standard k-fold cross-validation is fatally flawed for time series; chronological splits are critical to prevent future data from contaminating training.
3. **Domain Interaction Features**: Custom features like `Promo_DayOfWeek` ranked in the top 5 SHAP values, highlighting that promotional lift is day-dependent.
4. **Explainability as a First-Class Citizen**: Black-box models struggle with operational adoption; SHAP bridges the gap between data science and warehouse managers.

---

## 👨‍💻 Author

**Priyank Mishra**  
- GitHub: [@Number789Alpha](https://github.com/Number789Alpha)  
- Live App: [retail-demand-forecaster.streamlit.app](https://retail-demand-forecaster.streamlit.app)
