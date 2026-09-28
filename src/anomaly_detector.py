from collections import deque
from math import sqrt
from typing import Optional


class StreamingAnomalyDetector:
    """
    Online anomaly detector based on a rolling z-score.

    Each incoming value is compared against statistics from the
    previous window, then incorporated into the rolling window.

    Time complexity:
        update(): O(1)
        reset(): O(1)

    Space complexity:
        O(window_size)
    """

    def __init__(
        self,
        window_size: int = 100,
        z_threshold: float = 3.0,
        min_samples: int = 10,
    ) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1")

        if z_threshold <= 0:
            raise ValueError("z_threshold must be greater than 0")

        if min_samples < 2:
            raise ValueError("min_samples must be at least 2")

        if min_samples > window_size:
            raise ValueError("min_samples cannot exceed window_size")

        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_samples = min_samples

        self._window = deque(maxlen=window_size)

        # Running statistics.
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def count(self) -> int:
        return len(self._window)

    @property
    def mean(self) -> Optional[float]:
        if not self._window:
            return None
        return self._mean

    @property
    def variance(self) -> Optional[float]:
        n = len(self._window)

        if n < 2:
            return None

        return max(self._m2 / (n - 1), 0.0)

    @property
    def standard_deviation(self) -> Optional[float]:
        variance = self.variance

        if variance is None:
            return None

        return sqrt(variance)

    def _add_value(self, value: float) -> None:
        """Add a value using Welford's online algorithm."""
        n = len(self._window)

        if n == 0:
            self._window.append(value)
            self._mean = value
            self._m2 = 0.0
            return

        new_n = n + 1
        delta = value - self._mean
        self._mean += delta / new_n
        delta2 = value - self._mean
        self._m2 += delta * delta2

        self._window.append(value)

    def _remove_oldest(self) -> None:
        """Remove the oldest value while maintaining running statistics."""
        value = self._window.popleft()
        n = len(self._window)

        if n == 0:
            self._mean = 0.0
            self._m2 = 0.0
            return

        old_mean = self._mean
        new_mean = (old_mean * (n + 1) - value) / n

        self._m2 -= (value - old_mean) * (value - new_mean)
        self._m2 = max(self._m2, 0.0)

        self._mean = new_mean

    def update(self, x: Optional[float]) -> bool:
        """
        Process one incoming value.

        Returns True when the value is an anomaly, otherwise False.
        Invalid/missing values are ignored.
        """

        if x is None:
            return False

        try:
            value = float(x)
        except (TypeError, ValueError):
            return False

        if not (value == value):  # NaN check
            return False

        if value == float("inf") or value == float("-inf"):
            return False

        # Warm-up phase.
        if len(self._window) < self.min_samples:
            self._add_value(value)
            return False

        # Compare against the current window BEFORE adding the new value.
        standard_deviation = self.standard_deviation

        if standard_deviation is None or standard_deviation == 0:
            is_anomaly = False
        else:
            z_score = abs(value - self._mean) / standard_deviation
            is_anomaly = z_score > self.z_threshold

        # Maintain a bounded rolling window.
        if len(self._window) == self.window_size:
            self._remove_oldest()

        self._add_value(value)

        return is_anomaly

    def reset(self) -> None:
        """Clear the detector state."""
        self._window.clear()
        self._mean = 0.0
        self._m2 = 0.0