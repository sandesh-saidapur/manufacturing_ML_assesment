import pytest

from src.anomaly_detector import StreamingAnomalyDetector


def test_detector_stays_inactive_during_warmup():
    detector = StreamingAnomalyDetector(
        window_size=10,
        z_threshold=3.0,
        min_samples=5,
    )

    results = [
        detector.update(10.0),
        detector.update(10.1),
        detector.update(9.9),
        detector.update(10.0),
        detector.update(10.1),
    ]

    assert results == [
        False,
        False,
        False,
        False,
        False,
    ]

    assert detector.count == 5


def test_normal_values_are_not_anomalies():
    detector = StreamingAnomalyDetector(
        window_size=20,
        z_threshold=3.0,
        min_samples=5,
    )

    # Warm-up
    for value in [10.0, 10.1, 9.9, 10.0, 10.1]:
        detector.update(value)

    # Normal observation
    result = detector.update(10.05)

    assert result is False


def test_large_outlier_is_detected():
    detector = StreamingAnomalyDetector(
        window_size=20,
        z_threshold=3.0,
        min_samples=5,
    )

    # Build a stable baseline.
    for value in [10.0, 10.1, 9.9, 10.0, 10.1]:
        detector.update(value)

    # Very large deviation.
    result = detector.update(100.0)

    assert result is True


def test_missing_values_are_ignored():
    detector = StreamingAnomalyDetector(
        window_size=10,
        z_threshold=3.0,
        min_samples=5,
    )

    for value in [10.0, 10.1, 9.9, 10.0, 10.1]:
        detector.update(value)

    count_before = detector.count

    assert detector.update(None) is False
    assert detector.count == count_before

    assert detector.update(float("nan")) is False
    assert detector.count == count_before


def test_infinite_values_are_ignored():
    detector = StreamingAnomalyDetector(
        window_size=10,
        z_threshold=3.0,
        min_samples=5,
    )

    for value in [10.0, 10.1, 9.9, 10.0, 10.1]:
        detector.update(value)

    count_before = detector.count

    assert detector.update(float("inf")) is False
    assert detector.count == count_before

    assert detector.update(float("-inf")) is False
    assert detector.count == count_before


def test_zero_variance_does_not_crash():
    detector = StreamingAnomalyDetector(
        window_size=10,
        z_threshold=3.0,
        min_samples=5,
    )

    for _ in range(5):
        detector.update(10.0)

    # Standard deviation is zero.
    # Detector should handle this safely.
    result = detector.update(10.0)

    assert result is False


def test_window_size_is_bounded():
    detector = StreamingAnomalyDetector(
        window_size=5,
        z_threshold=3.0,
        min_samples=3,
    )

    for value in range(20):
        detector.update(float(value))

    assert detector.count == 5


def test_reset_clears_detector():
    detector = StreamingAnomalyDetector(
        window_size=10,
        z_threshold=3.0,
        min_samples=5,
    )

    for value in [10.0, 10.1, 9.9, 10.0, 10.1]:
        detector.update(value)

    assert detector.count == 5

    detector.reset()

    assert detector.count == 0
    assert detector.mean is None
    assert detector.variance is None
    assert detector.standard_deviation is None


def test_invalid_window_size_raises_error():
    with pytest.raises(ValueError):
        StreamingAnomalyDetector(
            window_size=1,
            z_threshold=3.0,
            min_samples=2,
        )


def test_invalid_threshold_raises_error():
    with pytest.raises(ValueError):
        StreamingAnomalyDetector(
            window_size=10,
            z_threshold=0,
            min_samples=5,
        )


def test_invalid_min_samples_raises_error():
    with pytest.raises(ValueError):
        StreamingAnomalyDetector(
            window_size=10,
            z_threshold=3.0,
            min_samples=1,
        )


def test_min_samples_cannot_exceed_window_size():
    with pytest.raises(ValueError):
        StreamingAnomalyDetector(
            window_size=5,
            z_threshold=3.0,
            min_samples=10,
        )

def test_rolling_statistics_are_correct():
    detector = StreamingAnomalyDetector(
        window_size=3,
        z_threshold=3.0,
        min_samples=2,
    )

    detector.update(1.0)
    detector.update(2.0)
    detector.update(3.0)

    assert detector.count == 3
    assert detector.mean == pytest.approx(2.0)
    assert detector.variance == pytest.approx(1.0)
    assert detector.standard_deviation == pytest.approx(1.0)


def test_statistics_update_when_window_slides():
    detector = StreamingAnomalyDetector(
        window_size=3,
        z_threshold=3.0,
        min_samples=2,
    )

    detector.update(1.0)
    detector.update(2.0)
    detector.update(3.0)

    # Window becomes [2, 3, 4]
    detector.update(4.0)

    assert detector.count == 3
    assert detector.mean == pytest.approx(3.0)
    assert detector.variance == pytest.approx(1.0)


def test_invalid_non_numeric_input_is_ignored():
    detector = StreamingAnomalyDetector(
        window_size=10,
        z_threshold=3.0,
        min_samples=3,
    )

    detector.update(10.0)
    detector.update(10.1)

    count_before = detector.count

    assert detector.update("invalid") is False
    assert detector.count == count_before        