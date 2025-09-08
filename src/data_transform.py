# Data transformation logic for mid-price prediction

import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # Example transformation: fill NA and scale prices
    df = df.fillna(method='ffill')
    if 'MidPrice' in df.columns:
        df['MidPrice'] = (df['MidPrice'] - df['MidPrice'].mean()) / df['MidPrice'].std()
    return df
