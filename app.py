import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import timedelta

# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecaster",
    page_icon="🛒",
    layout="wide"
)

# ─── Load model and data ───────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('models/lgbm_model.pkl')
    features = joblib.load('models/features.pkl')
    metrics  = joblib.load('models/metrics.pkl')
    return model, features, metrics

@st.cache_data
def load_data():
    df = pd.read_csv('models/train_clean.csv', parse_dates=['Date'])
    return df

model, FEATURES, metrics = load_model()
train = load_data()

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shop.png", width=60)
st.sidebar.title("⚙️ Controls")

store_id = st.sidebar.selectbox(
    "Select Store",
    options=sorted(train['Store'].unique()),
    index=0
)

horizon = st.sidebar.radio(
    "Forecast Horizon",
    options=[7, 14, 30],
    format_func=lambda x: f"{x} days",
    index=0
)

promo_on = st.sidebar.toggle("🏷️ Promo Active", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("LightGBM RMSE", f"€{metrics['lgbm_rmse']:,.0f}")
st.sidebar.metric("RMSPE", f"{metrics['lgbm_rmspe']:.1f}%")
st.sidebar.metric("vs Baseline", f"+{metrics['rmse_improvement']:.1f}% better")

# ─── Header ────────────────────────────────────────────────
st.title("🛒 Retail Demand Forecasting Dashboard")
st.caption("Hyper-local sales forecasting using LightGBM + Feature Engineering | Rossmann Store Sales")

# ─── KPI Cards ─────────────────────────────────────────────
store_data   = train[train['Store'] == store_id]
avg_sales    = store_data['Sales'].mean()
max_sales    = store_data['Sales'].max()
store_type   = store_data['StoreType'].iloc[0]
promo_avg    = store_data[store_data['Promo']==1]['Sales'].mean()
no_promo_avg = store_data[store_data['Promo']==0]['Sales'].mean()
store_promo_lift = ((promo_avg - no_promo_avg) / no_promo_avg * 100) if no_promo_avg > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Store",               f"#{store_id} (Type {store_type})")
col2.metric("Avg Daily Sales",     f"€{avg_sales:,.0f}")
col3.metric("Peak Sales Day",      f"€{max_sales:,.0f}")
col4.metric("Promo Lift (this store)", f"+{store_promo_lift:.1f}%")

st.markdown("---")

# ─── Forecast function ─────────────────────────────────────
def make_forecast(store_id, horizon, promo_on):
    store_hist   = train[train['Store'] == store_id].sort_values('Date')
    last_date    = store_hist['Date'].max()
    last_row     = store_hist.iloc[-1]
    recent_sales = store_hist['Sales'].values

    rows = []
    for i in range(1, horizon + 1):
        forecast_date = last_date + timedelta(days=i)
        lag1      = recent_sales[-1]  if len(recent_sales) >= 1  else 6000
        lag7      = recent_sales[-7]  if len(recent_sales) >= 7  else 6000
        lag14     = recent_sales[-14] if len(recent_sales) >= 14 else 6000
        lag30     = recent_sales[-30] if len(recent_sales) >= 30 else 6000
        roll7     = np.mean(recent_sales[-7:])  if len(recent_sales) >= 7  else 6000
        roll14    = np.mean(recent_sales[-14:]) if len(recent_sales) >= 14 else 6000
        roll_std7 = np.std(recent_sales[-7:])   if len(recent_sales) >= 7  else 500
        roll_max7 = np.max(recent_sales[-7:])   if len(recent_sales) >= 7  else 7000

        promo_val      = int(promo_on)
        store_type_enc = last_row['StoreType_enc']
        assortment_enc = last_row['Assortment_enc']
        dow            = forecast_date.dayofweek + 1

        row = {
            'Store':               store_id,
            'DayOfWeek':           dow,
            'Promo':               promo_val,
            'SchoolHoliday':       0,
            'StoreType_enc':       store_type_enc,
            'Assortment_enc':      assortment_enc,
            'CompetitionDistance': last_row['CompetitionDistance'],
            'Promo2':              last_row['Promo2'],
            'StateHoliday_enc':    0,
            'Year':                forecast_date.year,
            'Month':               forecast_date.month,
            'Day':                 forecast_date.day,
            'WeekOfYear':          forecast_date.isocalendar()[1],
            'DayOfYear':           forecast_date.timetuple().tm_yday,
            'IsWeekend':           int(dow >= 6),
            'IsMonthStart':        int(forecast_date.day <= 5),
            'IsMonthEnd':          int(forecast_date.day >= 25),
            'Sales_Lag1':          lag1,
            'Sales_Lag7':          lag7,
            'Sales_Lag14':         lag14,
            'Sales_Lag30':         lag30,
            'Rolling_Mean_7':      roll7,
            'Rolling_Mean_14':     roll14,
            'Rolling_Std_7':       roll_std7,
            'Rolling_Max_7':       roll_max7,
            'Promo_StoreType':     promo_val * store_type_enc,
            'Promo_DayOfWeek':     promo_val * dow,
        }
        rows.append(row)

        pred_df      = pd.DataFrame([row])[FEATURES]
        pred         = max(model.predict(pred_df)[0], 0)
        recent_sales = np.append(recent_sales, pred)

    forecast_df = pd.DataFrame(rows)
    preds       = np.maximum(model.predict(forecast_df[FEATURES]), 0)

    return pd.DataFrame({
        'Date':            [last_date + timedelta(days=i) for i in range(1, horizon+1)],
        'Predicted_Sales': preds,
        'Lower':           preds * 0.88,
        'Upper':           preds * 1.12
    })

forecast_df = make_forecast(store_id, horizon, promo_on)

# ─── Forecast Chart ────────────────────────────────────────
st.subheader(f"📈 {horizon}-Day Sales Forecast — Store #{store_id}"
             + (" 🏷️ Promo ON" if promo_on else ""))

hist = train[train['Store'] == store_id].sort_values('Date').tail(30)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist['Date'], y=hist['Sales'],
    mode='lines+markers',
    name='Historical Sales',
    line=dict(color='#4A90D9', width=2),
    marker=dict(size=4)
))

fig.add_trace(go.Scatter(
    x=pd.concat([forecast_df['Date'], forecast_df['Date'][::-1]]),
    y=pd.concat([forecast_df['Upper'], forecast_df['Lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(255,140,0,0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Band'
))

fig.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['Predicted_Sales'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#FF8C00', width=2.5, dash='dash'),
    marker=dict(size=6, symbol='diamond')
))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Sales (€)',
    hovermode='x unified',
    legend=dict(orientation='h', y=1.1),
    height=420,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ─── Forecast table ────────────────────────────────────────
with st.expander("📋 View Forecast Table"):
    display_df = forecast_df.copy()
    display_df['Date']            = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Predicted_Sales'] = display_df['Predicted_Sales'].apply(lambda x: f"€{x:,.0f}")
    display_df['Lower']           = display_df['Lower'].apply(lambda x: f"€{x:,.0f}")
    display_df['Upper']           = display_df['Upper'].apply(lambda x: f"€{x:,.0f}")
    display_df.columns            = ['Date','Predicted Sales','Lower Bound','Upper Bound']
    st.dataframe(display_df, use_container_width=True)

st.markdown("---")

# ─── Store Insights ────────────────────────────────────────
st.subheader("📊 Store Insights")

col1, col2 = st.columns(2)

with col1:
    # Reindex to ensure exactly 7 days always
    dow_sales = (store_data.groupby('DayOfWeek')['Sales']
                 .mean()
                 .reindex([1,2,3,4,5,6,7], fill_value=0))
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig2 = px.bar(
        x=days,
        y=dow_sales.values,
        title=f"Avg Sales by Day of Week — Store #{store_id}",
        color=dow_sales.values,
        color_continuous_scale='Blues',
        labels={'x':'Day', 'y':'Avg Sales (€)'}
    )
    fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    monthly       = store_data.groupby(store_data['Date'].dt.to_period('M'))['Sales'].mean()
    monthly.index = monthly.index.astype(str)
    fig3 = px.line(
        x=monthly.index,
        y=monthly.values,
        title=f"Monthly Sales Trend — Store #{store_id}",
        labels={'x':'Month', 'y':'Avg Sales (€)'},
        color_discrete_sequence=['#4A90D9']
    )
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ─── Footer ────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:grey; font-size:13px; padding:10px'>
    Built with LightGBM + Prophet + SHAP | 27 engineered features | 844K training rows<br>
    B.Tech CSE Final Year Project — Retail Demand Forecasting
</div>
""", unsafe_allow_html=True)