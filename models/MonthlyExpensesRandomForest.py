import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os

# Load dataset
data_path = 'C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/ME/CleanedExpenses.csv'
df = pd.read_csv(data_path)

# Check Financial_Health
if df['Financial_Health'].equals(df['Monthly_expenses_$']):
    print("Dropping Financial_Health (identical to Monthly_expenses_$)")
    df = df.drop('Financial_Health', axis=1)
else:
    print("Keeping Financial_Health")

# Features and target
X = df.drop('Monthly_expenses_$', axis=1)
y = df['Monthly_expenses_$']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_path = 'C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/Assets/ME/expenses_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved at: {scaler_path}")

print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Predict
y_pred_train = rf.predict(X_train_scaled)
y_pred_test = rf.predict(X_test_scaled)

# Evaluate
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Train MAE: {mae_train:.4f}")
print(f"Test MAE: {mae_test:.4f}")
print(f"Train R²: {r2_train:.4f}")
print(f"Test R²: {r2_test:.4f}")
print(f"Test RMSE: {rmse_test:.4f}")

# Cross-validation
cv_mae = -cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()
cv_r2 = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1).mean()
print(f"5-Fold CV MAE: {cv_mae:.4f}")
print(f"5-Fold CV R²: {cv_r2:.4f}")

# Feature importance
feature_names = X.columns.tolist()
importance = pd.DataFrame({'Feature': feature_names, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
print("\nTop 5 Features:\n", importance.head())
importance.to_csv('C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/models/rf_feature_importance.csv', index=False)

# Save model
model_path = 'C:/Users/anand/OneDrive/Documents/MoneyLit/MoneyLit/models/expenses_rf.joblib'
joblib.dump(rf, model_path)
print(f"Model saved at: {model_path}")

# Verify
if os.path.exists(model_path):
    loaded_model = joblib.load(model_path)
    test_pred = loaded_model.predict(X_test_scaled[:1])[0]
    print(f"Test prediction from loaded model: {test_pred:.4f}")
else:
    print("Error: Model file not found.")

# Dollar range
min_expenses, max_expenses = 0, 500

# Inverse-scale
y_pred_test_dollars = y_pred_test * (max_expenses - min_expenses) + min_expenses
y_test_dollars = y_test * (max_expenses - min_expenses) + min_expenses

# Evaluate in dollars
mae_test_dollars = mean_absolute_error(y_test_dollars, y_pred_test_dollars)
rmse_test_dollars = np.sqrt(mean_squared_error(y_test_dollars, y_pred_test_dollars))

print(f"Test MAE (Dollars): ${mae_test_dollars:.2f}")
print(f"Test RMSE (Dollars): ${rmse_test_dollars:.2f}")

# Sample prediction
sample_idx = 0
print(f"Sample Prediction (Normalized): {y_pred_test[sample_idx]:.4f}")
print(f"Sample Prediction (Dollars): ${y_pred_test_dollars[sample_idx]:.2f}")