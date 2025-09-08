# Feature engineering for mid-price prediction

import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Example: Add rolling mean as a feature
    if 'MidPrice' in df.columns:
        df['MidPrice_RollingMean'] = df['MidPrice'].rolling(window=5).mean()
    return df
