Retail Demand Forecasting System

An end-to-end Machine Learning + Data Science pipeline that predicts daily store-level sales — the kind of system that powers inventory decisions at Blinkit, Swiggy Instamart, and Flipkart Quick Commerce.

🔗 Live Demo
👉 retail-demand-forecaster.streamlit.app

📌 Problem Statement
Dark stores and retail chains lose crores daily due to poor inventory management:

❌ Overstock → products expire, capital wasted, storage costs rise
❌ Understock → orders fail, customers leave, revenue lost

This project builds a machine learning system that predicts daily demand per store using historical sales data, promotional signals, and seasonal patterns — enabling smarter, data-driven inventory decisions.

📊 Dataset
Rossmann Store Sales — Real historical sales data from Kaggle
MetricValueTotal rows (after cleaning)844,338Number of stores1,115Time periodJan 2013 – Jul 2015Original features9Engineered features27Target variableDaily Sales (€)

🏗️ Project Architecture
Raw Data (train.csv, store.csv)
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering (27 features)
        ↓
Model Training & Evaluation
   ├── Linear Regression (Baseline)
   ├── LightGBM (Primary Model)
   └── Prophet (Time-Series Decomposition)
        ↓
SHAP Explainability
        ↓
Streamlit Dashboard → Deployed on Streamlit Cloud

🔍 Exploratory Data Analysis
Key insights discovered through EDA:
InsightFindingPromo effect+38.8% higher sales on promotional daysBest sales dayMondayPeak monthDecember (+45%) — holiday season spikeBest store typeType b — avg €10,233 vs €6,900 for othersCompetition effectStores with competitor >5km away sell moreSales distributionLog transformation normalises the right-skewed distribution
EDA Visualisations

Sales distribution (normal vs log)
Average sales by day of week
Promo vs no-promo comparison
Monthly sales trend (2013–2015)
Seasonal decomposition (trend + seasonality + residual)
Correlation heatmap
Store type analysis
Competition distance effect


⚙️ Feature Engineering
Engineered 27 features from 9 original columns:
CategoryFeaturesPurposeLag FeaturesSales_Lag1, Lag7, Lag14, Lag30Capture sales momentum and autocorrelationRolling WindowRolling_Mean_7, Mean_14, Std_7, Max_7Smooth recent trendsDate FeaturesYear, Month, Day, WeekOfYear, DayOfYear, IsWeekend, IsMonthStart, IsMonthEndCapture time-based patternsInteractionPromo_StoreType, Promo_DayOfWeekPromo effect varies by store type and dayCategorical EncodingStoreType_enc, Assortment_enc, StateHoliday_encConvert categories to numeric

⚠️ Important: Time-based train/val split used (NOT random) to prevent data leakage. Last 6 weeks held out as validation.


🤖 Models Used
1. Linear Regression — Baseline

Establishes minimum performance benchmark
RMSE: €1,245 | RMSPE: 20.6%

2. LightGBM — Primary Model

Gradient boosting with 995 sequential decision trees
Handles non-linear patterns and feature interactions
Early stopping to prevent overfitting
RMSE: €851 | RMSPE: 12.6%
31.6% improvement over baseline

Key hyperparameters:
pythonLGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
3. Facebook Prophet — Time-Series Decomposition

Decomposes sales into trend + weekly + yearly seasonality
Confirms Monday peak and December +45% holiday spike
Provides confidence intervals for future forecasts


🧠 SHAP Explainability
Every prediction is explained using SHAP (SHapley Additive exPlanations):
RankFeatureSHAP ImportanceInsight1Rolling_Mean_14HighestRecent 2-week sales history matters most2PromoVery HighConfirms 38.8% EDA finding3Sales_Lag1HighYesterday strongly predicts today4DayOfWeekHighDay of week is a major driver5Promo_DayOfWeekMediumCustom interaction feature validated
Waterfall chart example: For a prediction of €6,137 (actual: €5,201):

Promo=0 pushed DOWN by €436
Sales_Lag1=5591 pushed DOWN by €376
Rolling_Mean_14=7296 pushed UP by €347


📈 Results Summary
ModelRMSERMSPEImprovementLinear Regression (Baseline)€1,24520.6%—LightGBM€85112.6%+31.6%

Training samples: 770,117
Validation samples: 58,611 (last 6 weeks)
Number of features: 27


🖥️ Dashboard Features
The live Streamlit dashboard includes:

Store selector — choose any of 1,115 stores
Forecast horizon — 7, 14, or 30 days
Promo toggle — see how activating a promo changes the forecast in real time
Confidence band — upper and lower prediction bounds
Forecast table — exact daily predictions with bounds
Store insights — day-of-week sales pattern + monthly trend
Model metrics — RMSE, RMSPE, improvement over baseline in sidebar


🛠️ Tech Stack
CategoryToolsLanguagePython 3.11ML ModelsLightGBM, Scikit-learnTime-SeriesFacebook Prophet, StatsmodelsExplainabilitySHAPData ProcessingPandas, NumPyVisualisationMatplotlib, Seaborn, PlotlyDashboardStreamlitDeploymentStreamlit Community CloudVersion ControlGit, GitHub

📁 Project Structure
retail-demand-forecaster/
│
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── models/
    ├── lgbm_model.pkl          # Trained LightGBM model
    ├── features.pkl            # Feature list
    ├── metrics.pkl             # Model performance metrics
    └── train_dashboard.csv     # Processed data for dashboard

🚀 Run Locally
bash# 1. Clone the repository
git clone https://github.com/Number789Alpha/retail-demand-forecaster.git
cd retail-demand-forecaster

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

Note: The full training notebook is available in Google Colab. The models/ folder contains all pre-trained artifacts needed to run the dashboard.


📓 Training Notebook
The complete ML pipeline (data cleaning → EDA → feature engineering → model training → SHAP) was built in Google Colab. Key notebook sections:

Blocks 1–5: Library imports and data loading
Blocks 6–10: Data cleaning and merging
Blocks 11–20: EDA and visualisations
Blocks 21–25: Feature engineering
Blocks 26–30: Model training and evaluation
Blocks 31–35: SHAP explainability and Prophet forecasting
Blocks 36–38: Model saving and final summary


💡 Key Learnings

A model is only as good as its features. The 31.6% improvement didn't come from a fancier algorithm — it came from understanding the data deeply first.


EDA before modelling — The 38.8% promo lift discovered in EDA became the 2nd most important SHAP feature
Time-based splits matter — Random splits in time-series cause data leakage and overestimate performance
Interaction features add value — Promo_DayOfWeek (a custom feature) made the top 5 SHAP features
Explainability builds trust — SHAP makes model predictions auditable and actionable for business teams
Feature engineering > algorithm choice — Lag and rolling features drove most of the accuracy gain


🎯 Business Impact
This system addresses a real operational challenge faced by every retail and quick commerce company:

Predicts daily sales with 12.6% average error — actionable for inventory planning
Promo toggle shows forecast impact before a promotion runs — helps plan promo ROI
Store-level granularity — 1,115 individual store forecasts, not just aggregate
Explainable predictions — every forecast comes with a reason via SHAP
