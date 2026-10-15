"""Trains the same Logistic Regression as the notebook and exposes a predict helper."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

CSV_PATH = "shopping.csv"


def train(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    # Same preprocessing as the notebook
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    raw_months = sorted(df["Month"].unique())
    raw_visitors = sorted(df["VisitorType"].unique())
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "buyer_recall": recall_score(y_test, pred),
        "buyer_precision": precision_score(y_test, pred, zero_division=0),
        "buyer_share": float(y.mean()),
    }
    return model, list(X.columns), metrics, raw_months, raw_visitors


def predict(model, columns, inputs: dict):
    """inputs uses the original CSV column names (Month / VisitorType as text)."""
    row = pd.DataFrame([inputs])
    row["Weekend"] = row["Weekend"].astype(int)
    row = pd.get_dummies(row)
    row = row.reindex(columns=columns, fill_value=0)  # missing dummy columns -> 0
    proba = float(model.predict_proba(row)[0, 1])
    return int(proba >= 0.5), proba
