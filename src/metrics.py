#!/usr/bin/env python3
from multiprocessing import Process, Semaphore
from util.import_export import load_metrics


def main():
    semaphore = Semaphore(8)
    processes = [
        Process(
            target=collect,
            args=(
                function,
                metric,
                dimension,
                semaphore,
            ),
        )
        for function in ["Rastrigin", "Griewangk", "Rosenbrock", "Michalewicz"]
        for dimension in [2, 30, 100]
        for metric in ["Fitness", "Runtime"]
    ]

    for process in processes:
        process.start()
        semaphore.acquire()

    for process in processes:
        process.join()


def collect(function: str, metric: str, dimensions: int, semaphore: Semaphore):
    try:
        metric_values_and_configurations = [
            (
                inertia_weight,
                cognitive_parameter,
                social_parameter,
                random_jitter_parameter,
                load_metrics(
                    f"PSO algorithm: {function}(dimensions = {dimensions}, inertia = {inertia_weight}, team_bias = {social_parameter}, cognitive_parameter = {cognitive_parameter}, random_jitter_parameter = {random_jitter_parameter}): {metric} - run 0"
                ),
            )
            for inertia_weight in [0.0, 0.25, 0.5, 0.75, 1.0]
            for cognitive_parameter in [0.0, 0.25, 0.5, 0.75, 1.0]
            for social_parameter in [0.0, 0.25, 0.5, 0.75, 1.0]
            for random_jitter_parameter in [0.0, 0.25, 0.5, 1, 2]
        ]
        metric_values_and_configurations = list(
            filter(lambda entry: len(entry[4]) > 90, metric_values_and_configurations)
        )
        metric_values_and_configurations = list(
            map(
                lambda entry: (
                    entry[0],
                    entry[1],
                    entry[2],
                    entry[3],
                    float(entry[4][-1][1]),
                ),
                metric_values_and_configurations,
            )
        )
        metric_values = list(
            map(
                lambda entry: entry[4],
                metric_values_and_configurations,
            )
        )
        metric_values.sort()

        if (len(metric_values) == 0):
            return

        best_metric_with_configuration = min(metric_values_and_configurations, key=lambda entry: entry[4])
        best_metric = min(metric_values)
        mean_metric = sum(metric_values) / len(metric_values)
        median_metric = metric_values[len(metric_values) // 2 + 1]
        worst_metric = max(metric_values)

        print(
            f"{function}: {metric} for {dimensions} dimensions:",
            f"best configuration: {best_metric_with_configuration}",
            f"best: {best_metric}",
            f"mean: {mean_metric}",
            f"median: {median_metric}",
            f"worst: {worst_metric}",
        )
    finally:
        semaphore.release()


if __name__ == "__main__":
    main()
