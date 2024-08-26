import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
date_range = pd.date_range(start="2010-01-01", end="2020-12-31", freq="D")

# Number of days
num_days = len(date_range)

# Generate random sales data
sales = np.random.randint(500, 5000, size=(num_days,))

# Create additional features
promotions = np.random.randint(0, 2, size=(num_days,))  # binary feature for promotions
holiday_flag = np.random.randint(0, 2, size=(num_days,))  # binary feature for holidays
marketing_spend = np.random.uniform(1000, 10000, size=(num_days,))
snowfall = np.random.uniform(0, 30, size=(num_days,))
temperature = np.random.uniform(-10, 35, size=(num_days,))
discounts = np.random.uniform(5, 30, size=(num_days,))
online_traffic = np.random.randint(1000, 10000, size=(num_days,))
store_traffic = np.random.randint(500, 5000, size=(num_days,))
ad_spend = np.random.uniform(500, 5000, size=(num_days,))
competitor_sales = np.random.uniform(400, 4000, size=(num_days,))
stock_level = np.random.uniform(100, 1000, size=(num_days,))
economic_index = np.random.uniform(80, 120, size=(num_days,))
seasonality = np.sin(np.linspace(0, 2 * np.pi, num_days)) * 1000 + 3000
trend = np.linspace(0, 1000, num_days)


data = pd.DataFrame(
    {
        "Date": date_range,
        "Sales": sales + seasonality + trend,
        "Promotions": promotions,
        "Holiday_Flag": holiday_flag,
        "Marketing_Spend": marketing_spend,
        "Snowfall": snowfall,
        "Temperature": temperature,
        "Discounts": discounts,
        "Online_Traffic": online_traffic,
        "Store_Traffic": store_traffic,
        "Ad_Spend": ad_spend,
        "Competitor_Sales": competitor_sales,
        "Stock_Level": stock_level,
        "Economic_Index": economic_index,
        "Seasonality": seasonality,
        "Trend": trend,
    }
)

# Introduce some missing values
data.loc[data.sample(frac=0.05).index, "Sales"] = np.nan  # 5% missing sales data

# Handle missing values by forward filling
data["Sales"].fillna(method="ffill", inplace=True)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.drop(columns=["Date"]))


# Create sequences for LSTM
def create_sequences(data, n_steps):
    sequences = []
    labels = []
    for i in range(len(data) - n_steps):
        sequences.append(data[i : i + n_steps, :])
        labels.append(data[i + n_steps, 0])  # Sales is the first column
    return np.array(sequences), np.array(labels)


n_steps = 30
sequences, labels = create_sequences(scaled_data, n_steps)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, shuffle=False
)

# Build and train LSTM Model
lstm_model = Sequential()
lstm_model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2]),
    )
)
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer="adam", loss="mean_squared_error")
lstm_model.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)
)

# Flatten the sequences for XGBoost
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train XGBoost Model
xgboost_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgboost_model.fit(X_train_flat, y_train)

# Predictions and evaluation
y_pred_lstm = lstm_model.predict(X_test)
y_pred_xgboost = xgboost_model.predict(X_test_flat)

# Inverse scaling for comparison
y_test_rescaled = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1)
)[:, 0]
y_pred_lstm_rescaled = scaler.inverse_transform(
    np.concatenate((y_pred_lstm, X_test[:, -1, 1:]), axis=1)
)[:, 0]
y_pred_xgboost_rescaled = scaler.inverse_transform(
    np.concatenate((y_pred_xgboost.reshape(-1, 1), X_test[:, -1, 1:]), axis=1)
)[:, 0]

# Calculate RMSE and MAE
rmse_lstm = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_lstm_rescaled))
mae_lstm = mean_absolute_error(y_test_rescaled, y_pred_lstm_rescaled)
rmse_xgboost = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_xgboost_rescaled))
mae_xgboost = mean_absolute_error(y_test_rescaled, y_pred_xgboost_rescaled)

print(f"LSTM Model - RMSE: {rmse_lstm}, MAE: {mae_lstm}")
print(f"XGBoost Model - RMSE: {rmse_xgboost}, MAE: {mae_xgboost}")

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label="Actual Sales")
plt.plot(y_pred_lstm_rescaled, label="LSTM Predictions")
plt.plot(y_pred_xgboost_rescaled, label="XGBoost Predictions")
plt.legend()
plt.title("Sales Forecasting - LSTM vs XGBoost")
plt.show()
