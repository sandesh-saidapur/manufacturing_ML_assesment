from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, load_data
from src.modeling import create_improved_model


RANDOM_STATE = 42


def evaluate(model, X, y, label):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    print(
        f"{label:<25} "
        f"F1={f1_score(y, predictions):.4f} | "
        f"ROC-AUC={roc_auc_score(y, probabilities):.4f}"
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "manufacturing_quality_dataset.csv"

    df = load_data(dataset_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = create_improved_model()
    model.fit(X_train, y_train)

    print("=" * 70)
    print("MISSINGNESS ROBUSTNESS")
    print("=" * 70)

    # Normal validation set.
    evaluate(model, X_val, y_val, "Normal validation")

    rng = np.random.default_rng(RANDOM_STATE)

    # Simulate progressively worse sensor availability.
    for missing_rate in [0.05, 0.10, 0.20, 0.30]:
        X_degraded = X_val.copy()

        for feature in FEATURE_COLUMNS:
            mask = rng.random(len(X_degraded)) < missing_rate
            X_degraded.loc[mask, feature] = np.nan

        evaluate(
            model,
            X_degraded,
            y_val,
            f"{missing_rate:.0%} missing/sensor",
        )


if __name__ == "__main__":
    main()