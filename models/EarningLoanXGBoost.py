import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#Load Data
df = pd.read_csv('C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/Earning and Loan/CleanedData.csv')

# Approximate CPI factors (to 2025 dollars)
cpi_factors = {
    2007: 1.50,  # $1 in 2007 = $1.50 in 2025
    2008: 1.46,
    2009: 1.47,
    2010: 1.44,
    2011: 1.40,
    2012: 1.37,
    2013: 1.35,
    2014: 1.33,
    2015: 1.33,
    2016: 1.31,
    2017: 1.29,
    2018: 1.26,
    2019: 1.24,
    2020: 1.13,
    2021: 1.08,
    2022: 1.00,
    2023: 0.96,
    2024: 0.93
}

# Adjust earnings
df['earnings_med_adjusted'] = df.apply(lambda x: x['earnings_med'] * cpi_factors.get(x['year'], 1.0), axis=1)

# Save
df.to_csv('C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/Earning and Loan/CleanedData.csv', index=False)
print(df[['inst_name', 'year', 'earnings_med', 'earnings_med_adjusted']].head())

# Features and Target
X = df.drop(['earnings_med', 'earnings_med_adjusted', 'inst_name'], axis=1)
y = df['earnings_med_adjusted']  # Use inflation-adjusted target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Base learner (same params)
params = {'objective': 'reg:squarederror', 'max_depth': 3, 'eta': 0.1, 'seed': 42}
base_model = xgb.train(params, dtrain, num_boost_round=1)
y_pred_base = base_model.predict(dtest)
mae_base = mean_absolute_error(y_test, y_pred_base)
print(f"MAE for base model (adjusted): ${mae_base:.2f}")
base_model.save_model('C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/Earning and Loan/base_model_adjusted.json')

# Residuals
y_pred_train_base = base_model.predict(dtrain)
residuals = y_train - y_pred_train_base
print("Sample Residuals (first 5):", residuals[:5])
print("Mean Absolute Residual:", np.abs(residuals).mean())
np.save('C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/Earning and Loan/residuals_adjusted.npy', residuals)

# Second tree
residuals = np.load('C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/Earning and Loan/residuals_adjusted.npy')
dtrain_residuals = xgb.DMatrix(X_train, label=residuals)
second_tree = xgb.train(params, dtrain_residuals, num_boost_round=1)
y_pred_test = base_model.predict(dtest) + second_tree.predict(dtest)
mae_second = mean_absolute_error(y_test, y_pred_test)
print(f"MAE after Second Tree (adjusted): ${mae_second:.2f}")
second_tree.save_model('C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/Earning and Loan/second_tree_adjusted.json')

# Full model
params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.1, 'seed': 42}
full_model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10, verbose_eval=10)
y_pred_full = full_model.predict(dtest)
mae_full = mean_absolute_error(y_test, y_pred_full)
r2_full = r2_score(y_test, y_pred_full)
print(f"Full Model MAE (adjusted): ${mae_full:.2f}")
print(f"Full Model R²: {r2_full:.4f}")
full_model.save_model('EarningLoanXGBModel.json')

# Tuned model
params = {'objective': 'reg:squarederror', 'max_depth': 7, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42}
full_model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'test')], early_stopping_rounds=5, verbose_eval=10)
y_pred_full = full_model.predict(dtest)
mae_full = mean_absolute_error(y_test, y_pred_full)
r2_full = r2_score(y_test, y_pred_full)
print(f"Tuned Model MAE (adjusted): ${mae_full:.2f}")
print(f"Tuned Model R²: {r2_full:.4f}")
full_model.save_model('C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/Earning and Loan/earnings_xgboost_tuned_adjusted.json')

# Lynchburg prediction
inst_name = 'University of Lynchburg'
if inst_name in df['inst_name'].values:
    row = df[df['inst_name'] == inst_name].drop(['earnings_med', 'earnings_med_adjusted', 'inst_name'], axis=1)
    dmatrix = xgb.DMatrix(row)
    pred = full_model.predict(dmatrix)[0]
    year = df[df['inst_name'] == inst_name]['year'].iloc[0]
    true_earnings = df[df['inst_name'] == inst_name]['earnings_med_adjusted'].iloc[0]
    print(f"Predicted Earnings for {inst_name}: ${pred:.2f} (2025 Adjusted)")
    print(f"True Earnings for {inst_name}: ${true_earnings:.2f} (2025 Adjusted, Year {year})")

   