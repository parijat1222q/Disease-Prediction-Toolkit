import pandas as pd
from scripts.preprocess import preprocess_data
from scripts.train_models import train_models

def test_preprocess():
    # Mock dataset
    df = pd.DataFrame({
        "age": [25, 35, 45],
        "cholesterol": [200, 240, 180],
        "disease": [0, 1, 0]
    })
    X, y = preprocess_data(df, "disease")
    assert X.shape[0] == 3
    assert len(y) == 3

def test_train_models():
    import numpy as np
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    models = train_models(X, y)
    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models
