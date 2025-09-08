import pandas as pd
import numpy as np

# ----- Step 1: Load the Encoded Dataset -----
file_path = 'engineered_data_encoded.csv'  # Make sure this file exists after encoding UPDATE_TYPE
try:
    df = pd.read_csv(file_path, parse_dates=['TIME_CREATED'])
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please run the encoding script first and verify the filename.")
    exit()

# Sort by TIME_CREATED to ensure chronological order
df.sort_values('TIME_CREATED', inplace=True)
df.reset_index(drop=True, inplace=True)
print('Dataset loaded and sorted by TIME_CREATED.')
print("First few rows:")
print(df.head())

# ----- Step 2: Compute the Average Time Interval Between Rows -----
df['TIME_DIFF'] = df['TIME_CREATED'].diff().dt.total_seconds()
avg_time_diff = df['TIME_DIFF'].dropna().mean()
print(f"Average time difference between rows (seconds): {avg_time_diff:.3f}")

# ----- Step 3: Define Forecast Horizons Based on Actual Time Intervals -----
# For example, if you want a 5-minute forecast and each row represents avg_time_diff seconds:
row_shift_30sec   = int(30 / avg_time_diff)
row_shift_1min   = int(60 / avg_time_diff)
row_shift_5min   = int(300 / avg_time_diff)
row_shift_15min  = int(900 / avg_time_diff)
row_shift_30min  = int(1800 / avg_time_diff)
row_shift_1hour  = int(3600 / avg_time_diff)

forecast_horizons = {
    '30sec': row_shift_30sec,
    '1min': row_shift_1min,
    '5min': row_shift_5min,
    '15min': row_shift_15min,
    '30min': row_shift_30min,
    '1hour': row_shift_1hour,
}

print("Forecast horizons (in number of rows to shift) defined based on actual time intervals:")
print(forecast_horizons)

# ----- Step 4: Create Shifted Target Columns for Future MID_PRICE -----
# For each defined horizon, create a new column that represents the MID_PRICE that occurs that many rows ahead.
for horizon_name, shift_steps in forecast_horizons.items():
    df[f"FUTURE_MID_PRICE_{horizon_name}"] = df["MID_PRICE"].shift(-shift_steps)

# ----- Step 5: Add Lagged Features to Capture Historical Dependencies -----
lag_list = [1, 2, 3]  # You can adjust these lags as necessary
for lag in lag_list:
    df[f"MID_PRICE_LAG_{lag}"] = df["MID_PRICE"].shift(lag)
    df[f"SPREAD_LAG_{lag}"] = df["SPREAD"].shift(lag)

# ----- Step 6: Drop Rows with NaN Values Resulting from Shifting Operations -----
df.dropna(inplace=True)
print("Final dataset shape after creating forecast targets and lag features:", df.shape)

# ----- Step 7: Save the Updated Dataset with Future Targets and Lag Features -----
output_file = 'future_midprice_with_lags_encoded.csv'
try:
    df.to_csv(output_file, index=False)
    print(f"Updated dataset saved as {output_file}.")
except Exception as e:
    print(f"Error saving file: {e}")
