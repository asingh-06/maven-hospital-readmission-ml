from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "start",
    train_fraction: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split data into train/test partitions."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    ordered = df.sort_values(date_col).reset_index(drop=True)
    split_idx = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def build_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> Dict[str, Pipeline]:
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    logistic = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Build a distinct preprocessor object for a separate pipeline.
    rf_preprocessor = make_preprocessor(numeric_features, categorical_features)
    random_forest = Pipeline(
        [
            ("preprocessor", rf_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
    }


def evaluate_binary_classifier(model, X_test, y_test, threshold: float = 0.50) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    if pd.Series(y_test).nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, probs)
        metrics["pr_auc"] = average_precision_score(y_test, probs)
    return metrics
