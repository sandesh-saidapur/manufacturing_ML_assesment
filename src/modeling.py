from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_COLUMNS = [
    "temperature_c",
    "pressure_bar",
    "vibration_mm_s",
    "humidity_pct",
]


def create_baseline_model() -> Pipeline:
    """
    Create a simple decision-tree baseline.

    Missing numerical values are median-imputed.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                FEATURE_COLUMNS,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=4,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def create_improved_model() -> HistGradientBoostingClassifier:
    """
    Create the improved nonlinear model.

    HistGradientBoosting handles missing numerical values
    natively and captures nonlinear relationships and
    feature interactions.
    """

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    return model