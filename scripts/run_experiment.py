from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import (
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    load_data,
    validate_columns,
    basic_data_summary,
)

from src.modeling import (
    create_baseline_model,
    create_improved_model,
)

from src.evaluation import (
    evaluate_classifier,
    print_metrics,
    calculate_psi,
)


RANDOM_STATE = 42

# Date separating historical training data from future data.
TEMPORAL_CUTOFF = pd.Timestamp("2025-02-15")


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a basic audit of the dataset.

    This includes:
    - dimensions
    - missing values
    - duplicates
    - target distribution
    - date range
    """

    summary = basic_data_summary(df)

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")

    print("\nMissing values:")

    for column, count in summary["missing_values"].items():

        percentage = count / len(df) * 100

        print(
            f"  {column}: "
            f"{count} ({percentage:.2f}%)"
        )

    print(
        f"\nDuplicate rows: "
        f"{summary['duplicates']}"
    )

    print("\nTarget distribution:")

    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nTarget percentage:")

    print(
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nDate range:")

    print(
        f"  {df['timestamp'].min()}"
    )

    print(
        f"  {df['timestamp'].max()}"
    )


def run_random_split_experiment(
    df: pd.DataFrame,
) -> None:
    """
    Train and evaluate the baseline and improved models
    using a stratified random train/validation split.
    """

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\n" + "=" * 60)
    print("RANDOM TRAIN / VALIDATION SPLIT")
    print("=" * 60)

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_val)}"
    )

    # --------------------------------------------------
    # Baseline model
    # --------------------------------------------------

    baseline = create_baseline_model()

    baseline.fit(
        X_train,
        y_train,
    )

    baseline_results = evaluate_classifier(
        baseline,
        X_val,
        y_val,
    )

    print_metrics(
        baseline_results,
        "Baseline: Decision Tree",
    )

    # --------------------------------------------------
    # Improved model
    # --------------------------------------------------

    improved = create_improved_model()

    improved.fit(
        X_train,
        y_train,
    )

    improved_results = evaluate_classifier(
        improved,
        X_val,
        y_val,
    )

    print_metrics(
        improved_results,
        "Improved: HistGradientBoosting",
    )


def calculate_distribution_shift(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> None:
    """
    Compare the training and future periods to identify
    distribution shift.

    We report:
    - target-rate change
    - feature mean changes
    - standardized mean shifts
    - PSI
    """

    print("\n" + "=" * 60)
    print("DISTRIBUTION SHIFT")
    print("=" * 60)

    # --------------------------------------------------
    # Target distribution shift
    # --------------------------------------------------

    print("\nTarget rate:")

    train_target_rate = (
        train_df[TARGET_COLUMN].mean()
    )

    future_target_rate = (
        future_df[TARGET_COLUMN].mean()
    )

    print(
        f"Training period: "
        f"{train_target_rate:.4f} "
        f"({train_target_rate * 100:.2f}%)"
    )

    print(
        f"Future period:   "
        f"{future_target_rate:.4f} "
        f"({future_target_rate * 100:.2f}%)"
    )

    print(
        f"Absolute change: "
        f"{future_target_rate - train_target_rate:+.4f}"
    )

    # --------------------------------------------------
    # Feature distribution shift
    # --------------------------------------------------

    print("\nFeature distribution shift:")

    for feature in FEATURE_COLUMNS:

        train_mean = (
            train_df[feature].mean()
        )

        future_mean = (
            future_df[feature].mean()
        )

        train_std = (
            train_df[feature].std()
        )

        mean_shift = (
            future_mean - train_mean
        )

        if train_std != 0:
            standardized_shift = (
                mean_shift / train_std
            )
        else:
            standardized_shift = 0.0

        psi = calculate_psi(
            train_df[feature],
            future_df[feature],
        )

        print(
            f"\n{feature}"
        )

        print(
            f"  Training mean: "
            f"{train_mean:.4f}"
        )

        print(
            f"  Future mean:   "
            f"{future_mean:.4f}"
        )

        print(
            f"  Mean change:   "
            f"{mean_shift:+.4f}"
        )

        print(
            f"  Standardized shift: "
            f"{standardized_shift:+.4f}"
        )

        print(
            f"  PSI: "
            f"{psi:.4f}"
        )


def run_temporal_experiment(
    df: pd.DataFrame,
) -> None:
    """
    Train on historical production data and evaluate
    on a later time period.

    This simulates deployment on future production data.
    """

    # Sort chronologically to make the temporal split explicit.
    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Temporal split
    # --------------------------------------------------

    train_df = df[
        df["timestamp"] < TEMPORAL_CUTOFF
    ].copy()

    future_df = df[
        df["timestamp"] >= TEMPORAL_CUTOFF
    ].copy()

    print("\n" + "=" * 60)
    print("TEMPORAL HOLDOUT")
    print("=" * 60)

    print(
        f"Training period: "
        f"{train_df['timestamp'].min().date()} "
        f"→ "
        f"{train_df['timestamp'].max().date()}"
    )

    print(
        f"Future period:   "
        f"{future_df['timestamp'].min().date()} "
        f"→ "
        f"{future_df['timestamp'].max().date()}"
    )

    print(
        f"\nTraining samples: "
        f"{len(train_df)}"
    )

    print(
        f"Future samples:   "
        f"{len(future_df)}"
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_future = future_df[FEATURE_COLUMNS]
    y_future = future_df[TARGET_COLUMN]

    # --------------------------------------------------
    # Baseline model
    # --------------------------------------------------

    baseline = create_baseline_model()

    baseline.fit(
        X_train,
        y_train,
    )

    baseline_results = evaluate_classifier(
        baseline,
        X_future,
        y_future,
    )

    print_metrics(
        baseline_results,
        "Temporal: Decision Tree",
    )

    # --------------------------------------------------
    # Improved model
    # --------------------------------------------------

    improved = create_improved_model()

    improved.fit(
        X_train,
        y_train,
    )

    improved_results = evaluate_classifier(
        improved,
        X_future,
        y_future,
    )

    print_metrics(
        improved_results,
        "Temporal: HistGradientBoosting",
    )

    # --------------------------------------------------
    # Distribution drift
    # --------------------------------------------------

    calculate_distribution_shift(
        train_df,
        future_df,
    )


def main() -> None:
    """
    Main experiment entry point.
    """

    # Find the project root regardless of where this
    # script is executed from.
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    dataset_path = (
        project_root
        / "data"
        / "manufacturing_quality_dataset.csv"
    )

    # --------------------------------------------------
    # Load and validate data
    # --------------------------------------------------

    df = load_data(
        dataset_path
    )

    validate_columns(df)

    # --------------------------------------------------
    # Dataset audit
    # --------------------------------------------------

    print_dataset_summary(df)

    # --------------------------------------------------
    # Random validation experiment
    # --------------------------------------------------

    run_random_split_experiment(
        df
    )

    # --------------------------------------------------
    # Temporal robustness experiment
    # --------------------------------------------------

    run_temporal_experiment(
        df
    )


if __name__ == "__main__":
    main()