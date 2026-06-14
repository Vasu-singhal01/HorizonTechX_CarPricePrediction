# ============================================================
#  Car Price Prediction Using Machine Learning
#  Horizon TechX Data Science Internship — Task 3
#  Dataset: https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars
# ============================================================

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'plotly', 'scikit-learn', '-q'])

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('car_data.csv')
df = df.drop_duplicates()
df['Car_Age'] = 2024 - df['Year']
le = LabelEncoder()
df['Fuel_Type_Enc']    = le.fit_transform(df['Fuel_Type'])
df['Selling_type_Enc'] = le.fit_transform(df['Selling_type'])
df['Transmission_Enc'] = le.fit_transform(df['Transmission'])

print("="*55)
print("        CAR PRICE DATASET — OVERVIEW")
print("="*55)
print(f"  Total Records  : {df.shape[0]}")
print(f"  Total Features : {df.shape[1]}")
print(f"  Missing Values : {df.isnull().sum().sum()}")
print("="*55)

features = ['Present_Price', 'Driven_kms', 'Car_Age', 'Owner',
            'Fuel_Type_Enc', 'Selling_type_Enc', 'Transmission_Enc']
X = df[features]
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

lr_r2   = r2_score(y_test, lr_pred)
rf_r2   = r2_score(y_test, rf_pred)
lr_mae  = mean_absolute_error(y_test, lr_pred)
rf_mae  = mean_absolute_error(y_test, rf_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"\n  Linear Regression → R²: {lr_r2*100:.2f}%  MAE: {lr_mae:.4f}  RMSE: {lr_rmse:.4f}")
print(f"  Random Forest     → R²: {rf_r2*100:.2f}%  MAE: {rf_mae:.4f}  RMSE: {rf_rmse:.4f}")
best_model = "Random Forest Regressor" if rf_r2 > lr_r2 else "Linear Regression"
print(f"\n🏆 Best Model: {best_model}")

BG=  '#0d1117'; CARD='#161b22'; BLUE='#58a6ff'
RED= '#f78166'; GREEN='#3fb950'; TEXT='#e6edf3'
ORG= '#ffa657'; PURP='#d2a8ff'

fig1 = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        '📊 Selling Price Distribution',
        '🔥 Present Price vs Selling Price',
        '⛽ Avg Price by Fuel Type',
        '⚙️ Manual vs Automatic Price',
        '📅 Car Age vs Selling Price',
        '🚗 Top 10 Cars by Avg Price'
    ),
    specs=[
        [{"type": "histogram"}, {"type": "scatter"}],
        [{"type": "bar"},       {"type": "bar"}],
        [{"type": "scatter"},   {"type": "bar"}]
    ],
    vertical_spacing=0.12, horizontal_spacing=0.10
)

fig1.add_trace(go.Histogram(x=df['Selling_Price'], nbinsx=30,
    marker_color=BLUE, opacity=0.8), row=1, col=1)

fig1.add_trace(go.Scatter(x=df['Present_Price'], y=df['Selling_Price'],
    mode='markers', marker=dict(color=df['Car_Age'],
    colorscale='Viridis', size=7, opacity=0.7)), row=1, col=2)

fuel_avg = df.groupby('Fuel_Type')['Selling_Price'].mean().reset_index()
fig1.add_trace(go.Bar(x=fuel_avg['Fuel_Type'], y=fuel_avg['Selling_Price'],
    marker_color=[GREEN, RED, ORG],
    text=[f'{v:.1f}L' for v in fuel_avg['Selling_Price']],
    textposition='outside'), row=2, col=1)

trans_avg = df.groupby('Transmission')['Selling_Price'].mean().reset_index()
fig1.add_trace(go.Bar(x=trans_avg['Transmission'], y=trans_avg['Selling_Price'],
    marker_color=[BLUE, PURP],
    text=[f'{v:.1f}L' for v in trans_avg['Selling_Price']],
    textposition='outside'), row=2, col=2)

fig1.add_trace(go.Scatter(x=df['Car_Age'], y=df['Selling_Price'],
    mode='markers', marker=dict(color=RED, size=6, opacity=0.6)), row=3, col=1)

top_cars = df.groupby('Car_Name')['Selling_Price'].mean().sort_values(ascending=False).head(10)
fig1.add_trace(go.Bar(x=top_cars.values[::-1], y=top_cars.index[::-1],
    orientation='h', marker_color=ORG,
    text=[f'{v:.1f}L' for v in top_cars.values[::-1]],
    textposition='outside'), row=3, col=2)

fig1.update_layout(
    title=dict(
        text='🚗 Car Price Prediction — EDA Dashboard<br><sup>Horizon TechX Data Science Internship | Task 3</sup>',
        font=dict(size=16, color=TEXT), x=0.5),
    height=1200, plot_bgcolor=CARD, paper_bgcolor=BG,
    font=dict(color=TEXT), showlegend=False
)
fig1.update_xaxes(gridcolor='#21262d')
fig1.update_yaxes(gridcolor='#21262d')
fig1.show()

feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '🏆 Model Accuracy (R² Score)',
        '📉 Error Comparison (MAE & RMSE)',
        '🎯 Actual vs Predicted (Random Forest)',
        '🔍 Feature Importance'
    ),
    specs=[
        [{"type": "bar"},     {"type": "bar"}],
        [{"type": "scatter"}, {"type": "bar"}]
    ],
    vertical_spacing=0.15, horizontal_spacing=0.12
)

fig2.add_trace(go.Bar(
    x=['Linear Regression', 'Random Forest'],
    y=[lr_r2*100, rf_r2*100],
    marker_color=[BLUE, GREEN],
    text=[f'{lr_r2*100:.1f}%', f'{rf_r2*100:.1f}%'],
    textposition='outside'
), row=1, col=1)

fig2.add_trace(go.Bar(
    x=['LR MAE', 'RF MAE', 'LR RMSE', 'RF RMSE'],
    y=[lr_mae, rf_mae, lr_rmse, rf_rmse],
    marker_color=[BLUE, GREEN, RED, ORG],
    text=[f'{lr_mae:.2f}', f'{rf_mae:.2f}', f'{lr_rmse:.2f}', f'{rf_rmse:.2f}'],
    textposition='outside'
), row=1, col=2)

fig2.add_trace(go.Scatter(
    x=list(y_test), y=list(rf_pred), mode='markers',
    marker=dict(color=GREEN, size=7, opacity=0.7)
), row=2, col=1)
fig2.add_trace(go.Scatter(
    x=[float(y_test.min()), float(y_test.max())],
    y=[float(y_test.min()), float(y_test.max())],
    mode='lines', line=dict(color=RED, dash='dash', width=2)
), row=2, col=1)

fig2.add_trace(go.Bar(
    x=feat_imp.values[::-1], y=feat_imp.index[::-1],
    orientation='h', marker_color=PURP,
    text=[f'{v:.3f}' for v in feat_imp.values[::-1]],
    textposition='outside'
), row=2, col=2)

fig2.update_layout(
    title=dict(
        text='🏆 Model Evaluation Dashboard — Car Price Prediction<br><sup>Horizon TechX Data Science Internship | Task 3</sup>',
        font=dict(size=16, color=TEXT), x=0.5),
    height=900, plot_bgcolor=CARD, paper_bgcolor=BG,
    font=dict(color=TEXT), showlegend=False
)
fig2.update_xaxes(gridcolor='#21262d')
fig2.update_yaxes(gridcolor='#21262d')
fig2.show()

print("\n✅ Car Price Prediction — All Complete!")
print(f"   Best Model : {best_model}")
print(f"   R² Score   : {rf_r2*100:.2f}%")
print(f"   MAE        : {rf_mae:.4f} Lakhs")
print(f"   RMSE       : {rf_rmse:.4f} Lakhs")