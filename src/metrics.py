#!/usr/bin/env python3
from multiprocessing import Process, Semaphore
from util.import_export import load_metrics


def main():
    semaphore = Semaphore(8)
    processes = [
        Process(
            target=collect_discrete,
            args=(
                algorithm,
                function,
                metric,
                dimension,
                semaphore,
            ),
        )
        for algorithm in [
            "Binary hillclimber algorithm",
            "Continuous hillclimber algorithm",
            "Binary genetic algorithm",
            "Hybrid algorithm",
        ]
        for function in ["Rastrigin", "Griewangk", "Rosenbrock", "Michalewicz"]
        for dimension in [2, 30, 100]
        for metric in ["Fitness", "Runtime"]
    ]

    for process in processes:
        process.start()
        semaphore.acquire()

    for process in processes:
        process.join()


def collect_discrete(
    algorithm: str, function: str, metric: str, dimensions: int, semaphore: Semaphore
):
    try:
        print(f"{algorithm}: {function}(dimensions = {dimensions}): {metric} - run 0")
        metric_values = load_metrics(
            f"{algorithm}: {function}(dimensions = {dimensions}): {metric} - run 0"
        )

        if len(metric_values) < 90:
            return

        metric_values = list(
            map(
                lambda entry: float(entry[1]),
                metric_values,
            )
        )
        metric_values.sort()

        best_metric = min(metric_values)
        mean_metric = sum(metric_values) / len(metric_values)
        median_metric = metric_values[len(metric_values) // 2 + 1]
        worst_metric = max(metric_values)

        print(
            f"{function}: {metric} for {dimensions} dimensions:",
            f"best: {best_metric}",
            f"mean: {mean_metric}",
            f"median: {median_metric}",
            f"worst: {worst_metric}",
        )
    finally:
        semaphore.release()


if __name__ == "__main__":
    main()
