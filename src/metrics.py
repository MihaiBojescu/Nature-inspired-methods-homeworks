#!/usr/bin/env python3
from dataclasses import dataclass
import os
import re
import typing as t
from itertools import chain
from multiprocessing import Process, Semaphore
from functions.combinatorial.tsp.util.common import MTSPResult
from util.sort import quicksort
from util.import_export import load_metrics

path = os.path.dirname(__file__)
property_regexp = re.compile(r"\b(\w+)\s*=\s*(\d+\.?\d*)\b")


class Runtime:
    value: float

    def __init__(self, value: str) -> None:
        self.value = float(value)

    def __eq__(self, other: t.Self) -> bool:
        return self.value == other.value

    def __le__(self, other: t.Self) -> bool:
        return self.value < other.value

    def __gt__(self, other: t.Self) -> bool:
        return self.value > other.value

    def __str__(self) -> str:
        return f"{self.value / 1000000000:.4f}"


def main():
    instances = build_instances()
    instances = wrap_variants_in_processes(instances, 10)

    instances()


def build_instances():
    return chain(
        iter(
            (
                collect_fitness,
                "APSO algorithm",
                dataset,
                dimensions,
                generations,
                population_size,
                probabilities,
                "Fitness",
            )
            for dataset in ["eil51", "berlin52", "eil76", "rat99"]
            for dimensions in [2, 3, 5]
            for generations in [200]
            for population_size in [100]
            for probabilities in [
                [("2-opt", 0.2), ("path-relink with pBest", 0.3), ("path-relink with gBest", 0.3), ("swap", 0.2)]
            ]
        ),
        iter(
            (
                collect_fitness,
                "Genetic algorithm",
                dataset,
                dimensions,
                generations,
                population_size,
                probabilities,
                "Fitness",
            )
            for dataset in ["eil51", "berlin52", "eil76", "rat99"]
            for dimensions in [2, 3, 5]
            for generations in [2000]
            for population_size in [100]
            for probabilities in [[("mutation", 0.1)]]
        ),
        iter(
            (
                collect_runtime,
                "APSO algorithm",
                dataset,
                dimensions,
                generations,
                population_size,
                probabilities,
                "Runtime",
            )
            for dataset in ["eil51", "berlin52", "eil76", "rat99"]
            for dimensions in [2, 3, 5]
            for generations in [200]
            for population_size in [100]
            for probabilities in [
                [("2-opt", 0.2), ("path-relink with pBest", 0.3), ("path-relink with gBest", 0.3), ("swap", 0.2)]
            ]
        ),
        iter(
            (
                collect_runtime,
                "Genetic algorithm",
                dataset,
                dimensions,
                generations,
                population_size,
                probabilities,
                "Runtime",
            )
            for dataset in ["eil51", "berlin52", "eil76", "rat99"]
            for dimensions in [2, 3, 5]
            for generations in [2000]
            for population_size in [100]
            for probabilities in [[("mutation", 0.1)]]
        ),
    )


def wrap_variants_in_processes(
    instances: t.List[t.Tuple[str, str, int, int, int, t.List[t.Tuple[str, float]], str]], concurrency: int
):
    def run():
        semaphore = Semaphore(concurrency)
        workers = []
        for (
            collect_method,
            algorithm,
            dataset,
            dimensions,
            generations,
            population_size,
            probabilities,
            metric,
        ) in instances:
            worker = Process(
                target=collect_method,
                args=(algorithm, dataset, dimensions, generations, population_size, probabilities, metric, semaphore),
            )
            worker.start()
            workers.append(worker)
            semaphore.acquire()

        for worker in workers:
            worker.join()

    return run


def collect_fitness(
    algorithm: str,
    dataset: str,
    dimensions: int,
    generations: int,
    population_size: int,
    probabilities: t.List[t.Tuple[str, int]],
    metric,
    semaphore: Semaphore,
):
    try:
        hyperparameters_string = ", ".join([f"{name} = {value}" for name, value in probabilities])
        name = f"{algorithm}: {dataset}(dimensions = {dimensions}, generations = {generations}, population_size = {population_size}, {hyperparameters_string})"

        metrics = [
            load_metrics(f"{name}: {metric} - run {run}", output_folder="outputs/discrete") for run in range(15)
        ]
        metrics = [
            [
                (
                    entry[0],
                    MTSPResult.from_data(entry[1]),
                )
                for entry in metric
            ]
            for metric in metrics
        ]
        metrics = [metric[-1][1] for metric in metrics]
        metrics = quicksort(metrics, comparator=lambda a, b: a < b)

        process_metrics(
            name=name,
            metrics=metrics,
        )
    finally:
        semaphore.release()


def collect_runtime(
    algorithm: str,
    dataset: str,
    dimensions: int,
    generations: int,
    population_size: int,
    probabilities: t.List[t.Tuple[str, int]],
    metric,
    semaphore: Semaphore,
):
    try:
        hyperparameters_string = ", ".join([f"{name} = {value}" for name, value in probabilities])
        name = f"{algorithm}: {dataset}(dimensions = {dimensions}, generations = {generations}, population_size = {population_size}, {hyperparameters_string}): {metric}"

        metrics = [
            load_metrics(f"{name} - run {run}", output_folder="outputs/discrete") for run in range(15)
        ]
        metrics = [
            [
                (
                    entry[0],
                    Runtime(value=entry[1]),
                )
                for entry in metric
            ]
            for metric in metrics
        ]
        metrics = [metric[-1][1] for metric in metrics]
        metrics = quicksort(metrics, comparator=lambda a, b: a < b)

        process_metrics(
            name=name,
            metrics=metrics,
        )
    finally:
        semaphore.release()


def process_metrics(
    name: str,
    metrics: t.List[any],
):
    if len(metrics) == 0:
        return

    best_metric = min(metrics)
    median_metric = metrics[len(metrics) // 2 + 1]
    worst_metric = max(metrics)

    print(
        name,
        f"best: {best_metric}",
        f"median: {median_metric}",
        f"worst: {worst_metric}",
    )


if __name__ == "__main__":
    main()
