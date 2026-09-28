import time

from src.anomaly_detector import StreamingAnomalyDetector


def benchmark(window_size: int, n_updates: int = 100_000) -> float:
    detector = StreamingAnomalyDetector(
        window_size=window_size,
        z_threshold=3.0,
        min_samples=min(10, window_size),
    )

    start = time.perf_counter()

    for i in range(n_updates):
        detector.update(float(i % 100))

    elapsed = time.perf_counter() - start
    updates_per_second = n_updates / elapsed

    print(
        f"window={window_size:5d} | "
        f"updates={n_updates:7d} | "
        f"time={elapsed:.4f}s | "
        f"updates/sec={updates_per_second:,.0f}"
    )

    return elapsed


def main() -> None:
    print("Streaming anomaly detector benchmark")
    print("-" * 70)

    for window_size in [10, 100, 1_000, 10_000]:
        benchmark(window_size)


if __name__ == "__main__":
    main()