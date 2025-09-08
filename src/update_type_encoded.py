import pandas as pd

# ----- Step 1: Load the Engineered Dataset -----
file_path = "engineered_data.csv"  # Ensure this file exists in your working directory

try:
    df = pd.read_csv(file_path, parse_dates=["TIME_CREATED"])
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path and try again.")
    exit()

# Sort by TIME_CREATED (if not already sorted)
df.sort_values("TIME_CREATED", inplace=True)
df.reset_index(drop=True, inplace=True)
print("Engineered dataset loaded and sorted by TIME_CREATED.")

# ----- Step 2: One-Hot Encode the UPDATE_TYPE Column -----
df_encoded = pd.get_dummies(df, columns=["UPDATE_TYPE"], prefix="TYPE")

# Display the first few rows to verify encoding
print("First few rows after encoding UPDATE_TYPE:")
print(df_encoded.head())

# ----- Step 3: Save the Encoded Dataset -----
output_file = "engineered_data_encoded.csv"
try:
    df_encoded.to_csv(output_file, index=False)
    print(f"Encoded dataset saved as '{output_file}'.")
except Exception as e:
    print(f"Error saving the file: {e}")
