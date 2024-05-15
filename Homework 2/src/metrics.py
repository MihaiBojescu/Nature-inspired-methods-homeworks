#!/usr/bin/env python3
import os
import re
import typing as t
from itertools import chain
from multiprocessing import Process, Semaphore
from util.import_export import load_metrics

path = os.path.dirname(__file__)
property_regexp = re.compile(r"\b(\w+)\s*=\s*(\d+\.?\d*)\b")


def main():
    instances = build_instances()
    instances = wrap_variants_in_processes(instances, 8)

    instances()


def build_instances():
    return chain(
        # build_discrete_instances(),
        build_stochastic_instances()
    )


def build_discrete_instances():
    return iter(
        (function, metric, dimension, "discrete")
        for function in ["Rastrigin", "Griewangk", "Rosenbrock", "Michalewicz"]
        for dimension in [2, 30, 100]
        for metric in ["Fitness", "Runtime"]
    )


def build_stochastic_instances():
    return iter(
        (function, metric, dimension, "stochastic")
        for function in ["Rastrigin", "Griewangk", "Rosenbrock", "Michalewicz"]
        for dimension in [2, 30, 100]
        for metric in ["Fitness", "Runtime"]
    )


def wrap_variants_in_processes(
    instances: t.List[t.Tuple[str, int, str]], concurrency: int
):
    def run():
        semaphore = Semaphore(concurrency)
        workers = []
        for function, metric, dimension, method in instances:
            worker = Process(
                target=collect_discrete if method == "discrete" else collect_stochastic,
                args=(
                    function,
                    metric,
                    dimension,
                    semaphore,
                ),
            )
            worker.start()
            workers.append(worker)
            semaphore.acquire()

        for worker in workers:
            worker.join()

    return run


def collect_discrete(function: str, metric: str, dimensions: int, semaphore: Semaphore):
    try:
        metric_values_and_configurations = [
            (
                (
                    generations,
                    inertia_weight,
                    cognitive_parameter,
                    social_parameter,
                    random_jitter_parameter,
                ),
                load_metrics(
                    f"PSO algorithm: {function}(dimensions = {dimensions}, inertia = {inertia_weight}, team_bias = {social_parameter}, cognitive_parameter = {cognitive_parameter}, random_jitter_parameter = {random_jitter_parameter}): {metric} - run 0"
                ),
            )
            for generations in [100]
            for inertia_weight in [0.0, 0.25, 0.5, 0.75, 1.0]
            for cognitive_parameter in [0.0, 0.25, 0.5, 0.75, 1.0]
            for social_parameter in [0.0, 0.25, 0.5, 0.75, 1.0]
            for random_jitter_parameter in [0.0, 0.25, 0.5, 1, 2]
        ]

        common_metrics_processing(
            function=function,
            metric=metric,
            dimensions=dimensions,
            metric_values_and_configurations=metric_values_and_configurations,
        )
    finally:
        semaphore.release()


def collect_stochastic(
    function: str, metric: str, dimensions: int, semaphore: Semaphore
):
    try:
        file_prefix = re.compile(
            rf"PSO algorithm: {function}\(dimensions = {dimensions},.+?\): {metric} - run 0"
        )
        files = os.listdir(os.path.join(path, "../outputs/stochastic"))
        files = list(map(lambda file: file.removesuffix(".csv"), files))
        files = list(filter(lambda file: file_prefix.search(file) != None, files))

        metric_values_and_configurations = [
            (
                extract_features(file),
                load_metrics(file, output_folder="outputs/stochastic"),
            )
            for file in files
        ]

        common_metrics_processing(
            function=function,
            metric=metric,
            dimensions=dimensions,
            metric_values_and_configurations=metric_values_and_configurations,
        )
    finally:
        semaphore.release()


def extract_features(file: str):
    features = property_regexp.findall(file)
    features = list(
        filter(
            lambda feature: feature[0]
            in [
                "generations",
                "inertia",
                "team_bias",
                "cognitive_parameter",
                "random_jitter_parameter",
            ],
            features,
        )
    )
    features = list(map(lambda feature: float(feature[1]), features))

    return tuple(features)


def common_metrics_processing(
    function: str,
    metric: str,
    dimensions: int,
    metric_values_and_configurations: t.List[t.Tuple[int, any]],
):
    metric_values_and_configurations = list(
        filter(lambda entry: len(entry[1]) > 90, metric_values_and_configurations)
    )
    metric_values_and_configurations = list(
        map(
            lambda entry: (
                entry[0],
                float(entry[1][-1][1]),
            ),
            metric_values_and_configurations,
        )
    )
    metric_values = list(
        map(
            lambda entry: entry[1],
            metric_values_and_configurations,
        )
    )
    metric_values.sort()

    if len(metric_values) == 0:
        return

    best_metric_with_configuration = min(
        metric_values_and_configurations, key=lambda entry: entry[1]
    )
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


if __name__ == "__main__":
    main()
