from typing import Any

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Evaluate a binary classification model.

    Returns a dictionary containing multiple complementary
    classification metrics.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    results = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "confusion_matrix": confusion_matrix(y, predictions),
    }

    return results


def print_metrics(
    results: dict,
    model_name: str,
) -> None:

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1       : {results['f1']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    The reference distribution represents the training period,
    while current represents a future/production period.

    Higher PSI indicates greater distributional change.
    """

    reference = reference.dropna()
    current = current.dropna()

    # Use quantiles from the reference distribution
    # to define stable bins.
    quantiles = np.linspace(0, 1, bins + 1)

    breakpoints = np.unique(
        reference.quantile(quantiles).values
    )

    # If there are not enough unique values,
    # fall back to equal-width bins.
    if len(breakpoints) < 3:
        breakpoints = np.linspace(
            reference.min(),
            reference.max(),
            bins + 1,
        )

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints,
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints,
    )

    # Add a small epsilon to prevent division by zero.
    epsilon = 1e-6

    reference_proportions = (
        reference_counts / len(reference)
    ) + epsilon

    current_proportions = (
        current_counts / len(current)
    ) + epsilon

    psi = np.sum(
        (
            current_proportions
            - reference_proportions
        )
        * np.log(
            current_proportions
            / reference_proportions
        )
    )

    return float(psi)