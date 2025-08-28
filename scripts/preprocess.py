import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col):
    # Drop rows with missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
