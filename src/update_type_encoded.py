# Update type encoding for mid-price prediction

import pandas as pd

def encode_update_type(df: pd.DataFrame) -> pd.DataFrame:
    # Example: One-hot encode 'UpdateType' column
    if 'UpdateType' in df.columns:
        update_type_dummies = pd.get_dummies(df['UpdateType'], prefix='UpdateType')
        df = pd.concat([df, update_type_dummies], axis=1)
        df = df.drop('UpdateType', axis=1)
    return df
