import pandas as pd
from sklearn.cluster import KMeans

def fit_regimes(X: pd.DataFrame, n_regimes: int, random_state: int = 42) -> KMeans:
    """Fit KMeans on feature matrix X to identify market regimes."""
    # random_state for reproducibility
    km = KMeans(n_clusters=n_regimes, n_init="auto", random_state=random_state)
    km.fit(X)
    return km

def label_regimes(X: pd.DataFrame, model: KMeans) -> pd.Series:
    """Predict regime labels for each row in X using the fitted model."""
    labels = model.predict(X)
    return pd.Series(labels, index=X.index, name="regime")
