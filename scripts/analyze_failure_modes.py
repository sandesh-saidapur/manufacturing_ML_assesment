from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "temperature_c",
    "pressure_bar",
    "vibration_mm_s",
    "humidity_pct",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "manufacturing_quality_dataset.csv"

    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])

    print("=" * 70)
    print("FAILURE MODE / BIAS ANALYSIS")
    print("=" * 70)

    # 1. Missingness vs target
    print("\n1. TARGET RATE BY MISSINGNESS")
    for feature in FEATURE_COLUMNS:
        missing = df[feature].isna()

        if missing.any():
            missing_rate = df.loc[missing, "defect"].mean()
            present_rate = df.loc[~missing, "defect"].mean()

            print(f"\n{feature}")
            print(f"  Missing observations: {missing.sum()}")
            print(f"  Defect rate when missing:  {missing_rate:.4f}")
            print(f"  Defect rate when present:  {present_rate:.4f}")

    # 2. Feature relationship with target
    print("\n2. FEATURE CORRELATION WITH TARGET")
    correlations = (
        df[FEATURE_COLUMNS + ["defect"]]
        .corr()["defect"]
        .drop("defect")
        .sort_values(key=abs, ascending=False)
    )

    print(correlations)

    # 3. Feature means by target
    print("\n3. FEATURE MEANS BY TARGET")
    print(
        df.groupby("defect")[FEATURE_COLUMNS]
        .mean()
        .round(4)
    )

    # 4. Temporal target-rate changes
    print("\n4. WEEKLY TARGET RATE")

    weekly = (
        df.assign(week=df["timestamp"].dt.to_period("W"))
        .groupby("week")["defect"]
        .agg(["mean", "count"])
    )

    weekly["mean"] = weekly["mean"].round(4)
    print(weekly)


if __name__ == "__main__":
    main()