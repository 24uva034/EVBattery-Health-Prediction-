import pandas as pd
import numpy as np

# Load actual EV battery dataset
df = pd.read_csv("archive.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst  5 rows:")
print(df.head())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nColumn Names:")
print(df.columns.tolist())

# ==============================
# STEP 2: DATA PREPROCESSING
# ==============================

# Separate numeric and categorical columns
numeric_columns = df.select_dtypes(include=np.number).columns
categorical_columns = df.select_dtypes(exclude=np.number).columns

# Fill missing numerical values with median
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
for col in categorical_columns:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# Check missing values after preprocessing
print("\nAfter Preprocessing:")
print("Total Missing Values:", df.isnull().sum().sum())

print("Dataset Shape:", df.shape)

# ==============================
# STEP 3: FEATURE EXTRACTION
# ==============================

# Create feature aliases based on project methodology
df["capacity_fade"] = df["capacity_loss_percent"]
df["internal_resistance_feature"] = df["internal_resistance"]

# Calculate voltage imbalance
df["voltage_imbalance"] = (
    df["cell_voltage_avg"] - df["pack_voltage"] / 100
)

# Select features for machine learning
features = [
    "cell_voltage_avg",
    "cell_voltage_std",
    "pack_voltage",
    "voltage_imbalance",
    "cycle_count",
    "capacity_fade",
    "internal_resistance_feature"
]

# Target variables
target_soh = "state_of_health"
target_rul = "predicted_remaining_life_cycles"

X = df[features]
y_soh = df[target_soh]
y_rul = df[target_rul]

print("\nFeature Extraction Completed!")
print("Features:", features)
print("Input Shape:", X.shape)
print("SOH Shape:", y_soh.shape)
print("RUL Shape:", y_rul.shape)

# ==============================
# STEP 4: TRAIN-TEST SPLIT
# AND NORMALIZATION
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Split data into training and testing sets
X_train, X_test, y_soh_train, y_soh_test, y_rul_train, y_rul_test = train_test_split(
    X,
    y_soh,
    y_rul,
    test_size=0.20,
    random_state=42
)

# Normalize the features
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTrain-Test Split Completed!")
print("Training Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

# ==============================
# STEP 5: RANDOM FOREST
# ==============================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Random Forest for SOH
rf_soh = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_soh.fit(X_train_scaled, y_soh_train)

# Predict SOH
rf_soh_pred = rf_soh.predict(X_test_scaled)

# Random Forest for RUL
rf_rul = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_rul.fit(X_train_scaled, y_rul_train)

# Predict RUL
rf_rul_pred = rf_rul.predict(X_test_scaled)

# SOH metrics
rf_soh_rmse = np.sqrt(mean_squared_error(y_soh_test, rf_soh_pred))
rf_soh_mae = mean_absolute_error(y_soh_test, rf_soh_pred)
rf_soh_r2 = r2_score(y_soh_test, rf_soh_pred)

# RUL metrics
rf_rul_rmse = np.sqrt(mean_squared_error(y_rul_test, rf_rul_pred))
rf_rul_mae = mean_absolute_error(y_rul_test, rf_rul_pred)
rf_rul_r2 = r2_score(y_rul_test, rf_rul_pred)

print("\n===== RANDOM FOREST RESULTS =====")

print("\nSOH:")
print("RMSE:", rf_soh_rmse)
print("MAE :", rf_soh_mae)
print("R2  :", rf_soh_r2)

print("\nRUL:")
print("RMSE:", rf_rul_rmse)
print("MAE :", rf_rul_mae)
print("R2  :", rf_rul_r2)

# ==============================
# STEP 6: SVR
# ==============================

from sklearn.svm import SVR

# SVR for SOH
svr_soh = SVR(kernel="rbf")

svr_soh.fit(X_train_scaled, y_soh_train)

# Predict SOH
svr_soh_pred = svr_soh.predict(X_test_scaled)

# SVR for RUL
svr_rul = SVR(kernel="rbf")

svr_rul.fit(X_train_scaled, y_rul_train)

# Predict RUL
svr_rul_pred = svr_rul.predict(X_test_scaled)

# SOH metrics
svr_soh_rmse = np.sqrt(mean_squared_error(y_soh_test, svr_soh_pred))
svr_soh_mae = mean_absolute_error(y_soh_test, svr_soh_pred)
svr_soh_r2 = r2_score(y_soh_test, svr_soh_pred)

# RUL metrics
svr_rul_rmse = np.sqrt(mean_squared_error(y_rul_test, svr_rul_pred))
svr_rul_mae = mean_absolute_error(y_rul_test, svr_rul_pred)
svr_rul_r2 = r2_score(y_rul_test, svr_rul_pred)

print("\n===== SVR RESULTS =====")

print("\nSOH:")
print("RMSE:", svr_soh_rmse)
print("MAE :", svr_soh_mae)
print("R2  :", svr_soh_r2)

print("\nRUL:")
print("RMSE:", svr_rul_rmse)
print("MAE :", svr_rul_mae)
print("R2  :", svr_rul_r2)

# ==============================
# STEP 7: GRADIENT BOOSTING
# ==============================

from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting for SOH
gb_soh = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

gb_soh.fit(X_train_scaled, y_soh_train)

# Predict SOH
gb_soh_pred = gb_soh.predict(X_test_scaled)

# Gradient Boosting for RUL
gb_rul = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

gb_rul.fit(X_train_scaled, y_rul_train)

# Predict RUL
gb_rul_pred = gb_rul.predict(X_test_scaled)

# SOH metrics
gb_soh_rmse = np.sqrt(mean_squared_error(y_soh_test, gb_soh_pred))
gb_soh_mae = mean_absolute_error(y_soh_test, gb_soh_pred)
gb_soh_r2 = r2_score(y_soh_test, gb_soh_pred)

# RUL metrics
gb_rul_rmse = np.sqrt(mean_squared_error(y_rul_test, gb_rul_pred))
gb_rul_mae = mean_absolute_error(y_rul_test, gb_rul_pred)
gb_rul_r2 = r2_score(y_rul_test, gb_rul_pred)

print("\n===== GRADIENT BOOSTING RESULTS =====")

print("\nSOH:")
print("RMSE:", gb_soh_rmse)
print("MAE :", gb_soh_mae)
print("R2  :", gb_soh_r2)

print("\nRUL:")
print("RMSE:", gb_rul_rmse)
print("MAE :", gb_rul_mae)
print("R2  :", gb_rul_r2)

# ==============================
# STEP 8: SAVE BEST MODELS
# ==============================

import pickle

# Save Gradient Boosting models
with open("gb_soh_model.pkl", "wb") as f:
    pickle.dump(gb_soh, f)

with open("gb_rul_model.pkl", "wb") as f:
    pickle.dump(gb_rul, f)

# Save scaler
with open("battery_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n===== MODEL SAVING =====")
print("SOH model saved: gb_soh_model.pkl")
print("RUL model saved: gb_rul_model.pkl")
print("Scaler saved: battery_scaler.pkl")