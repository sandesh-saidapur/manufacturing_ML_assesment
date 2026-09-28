from pathlib import Path
from typing import Union

import pandas as pd


TARGET_COLUMN = "defect"
TIMESTAMP_COLUMN = "timestamp"

FEATURE_COLUMNS = [
    "temperature_c",
    "pressure_bar",
    "vibration_mm_s",
    "humidity_pct",
]


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = (
        [TIMESTAMP_COLUMN]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def basic_data_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict(),
    }