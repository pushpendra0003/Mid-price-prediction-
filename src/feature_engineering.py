import pandas as pd
import numpy as np

# ----- Step 1: Load and Sort the Dataset -----
file_path = "combined_data.csv"  # Ensure this file is in your working directory
try:
    df = pd.read_csv(file_path, parse_dates=["TIME_CREATED"], infer_datetime_format=True)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path and try again.")
    exit()

# Convert TIME_CREATED explicitly if it's not already a datetime object.
# Here we assume the values are in milliseconds.
if not pd.api.types.is_datetime64_any_dtype(df["TIME_CREATED"]):
    df["TIME_CREATED"] = pd.to_datetime(df["TIME_CREATED"], unit='ms', origin='unix')

# Sort by TIME_CREATED to ensure chronological order.
df.sort_values("TIME_CREATED", inplace=True)
df.reset_index(drop=True, inplace=True)

print("Dataset loaded successfully!")
print(df.head())

# ----- Step 2: Feature Engineering -----

# 2.1 Volume Imbalance
print("Computing volume imbalance...")
df['ASK_VOLUME_TOP3'] = df[['ASK_1_QUANTITY', 'ASK_2_QUANTITY', 'ASK_3_QUANTITY']].sum(axis=1)
df['BID_VOLUME_TOP3'] = df[['BID_1_QUANTITY', 'BID_2_QUANTITY', 'BID_3_QUANTITY']].sum(axis=1)

# Avoid division by zero if the sum is zero.
df['VOLUME_IMBALANCE'] = (df['ASK_VOLUME_TOP3'] - df['BID_VOLUME_TOP3']) / \
                         (df['ASK_VOLUME_TOP3'] + df['BID_VOLUME_TOP3'] + 1e-6)

# 2.2 Price Imbalance & Derived Price Features
print("Computing price imbalance...")
# Compute MID_PRICE as the average of best ask and best bid.
df["MID_PRICE"] = (df["ASK_1_PRICE"] + df["BID_1_PRICE"]) / 2

# Compute SPREAD as the difference between best ask and bid.
df["SPREAD"] = df["ASK_1_PRICE"] - df["BID_1_PRICE"]

# Compute VWAP (Volume Weighted Average Price) for top 3 ask levels.
df['VWAP_ASK_TOP3'] = (
    df['ASK_1_PRICE'] * df['ASK_1_QUANTITY'] +
    df['ASK_2_PRICE'] * df['ASK_2_QUANTITY'] +
    df['ASK_3_PRICE'] * df['ASK_3_QUANTITY']
) / (df['ASK_VOLUME_TOP3'] + 1e-6)

# Compute VWAP for top 3 bid levels.
df['VWAP_BID_TOP3'] = (
    df['BID_1_PRICE'] * df['BID_1_QUANTITY'] +
    df['BID_2_PRICE'] * df['BID_2_QUANTITY'] +
    df['BID_3_PRICE'] * df['BID_3_QUANTITY']
) / (df['BID_VOLUME_TOP3'] + 1e-6)

# Compute price imbalance relative to MID_PRICE.
df['PRICE_IMBALANCE'] = (df['MID_PRICE'] - ((df['VWAP_BID_TOP3'] + df['VWAP_ASK_TOP3']) / 2)) / (df['MID_PRICE'] + 1e-6)

# 2.3 Time-Based Features
print("Extracting time-based features...")
if pd.api.types.is_datetime64_any_dtype(df["TIME_CREATED"]):
    df["HOUR_OF_DAY"] = df["TIME_CREATED"].dt.hour
    df["MINUTE_OF_DAY"] = df["TIME_CREATED"].dt.minute
else:
    print("Warning: TIME_CREATED is not a datetime object. Time-based features were not created.")

# 2.4 Exponential Moving Averages (EMAs)
print("Computing exponential moving averages...")
ema_periods = [5, 10, 50]
for period in ema_periods:
    # EMA for MID_PRICE (make sure MID_PRICE is computed before using)
    df[f'MID_PRICE_EMA_{period}'] = df["MID_PRICE"].ewm(span=period, adjust=False).mean()
    # EMA for SPREAD
    df[f'SPREAD_EMA_{period}'] = df["SPREAD"].ewm(span=period, adjust=False).mean()

# ----- Save Engineered Data -----
output_file = "engineered_data.csv"
try:
    df.to_csv(output_file, index=False)
    print(f"Feature engineering complete! Data saved as '{output_file}'.")
except Exception as e:
    print(f"Error saving file: {e}")
