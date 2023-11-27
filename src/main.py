#!/usr/bin/env python3
import typing as t
from multiprocessing import Process
from algorithms.pso.metered_algorithm import MeteredParticleSwarmOptimisation
from algorithms.base.algorithm import BaseAlgorithm
from functions.rastrigin import rastrigin_definition
from functions.michalewicz import michalewicz_definition
from functions.griewangk import griewangk_definition
from functions.rosenbrock import rosenbrock_definition
from functions.definition import FunctionDefinition
from util.import_export import save_metrics


def main():
    instances = build_instances()
    instances = wrap_instances_in_processes(instances)

    for algorithm in instances:
        algorithm.start()

    for algorithm in instances:
        algorithm.join()


def build_instances():
    return [
        (
            function_definition,
            dimension,
            inertia,
            MeteredParticleSwarmOptimisation.from_function_definition(
                function_definition=function_definition,
                dimensions=dimension,
                generations=500,
                inertia_bias=inertia,
            ),
        )
        for dimension in [2, 30, 100]
        for inertia in [0.2, 0.5, 1.0]
        for function_definition in [
            rosenbrock_definition,
            michalewicz_definition,
            griewangk_definition,
            rastrigin_definition,
        ]
    ]


def wrap_instances_in_processes(
    instances: t.List[t.Tuple[FunctionDefinition, int, float, BaseAlgorithm]]
):
    return [
        Process(
            target=process, args=(function_definition, dimensions, inertia, algorithm)
        )
        for (function_definition, dimensions, inertia, algorithm) in instances
    ]


def process(
    function_definition: FunctionDefinition,
    dimensions: int,
    inertia: float,
    algorithm: BaseAlgorithm,
):
    name = f"{algorithm.name}: {function_definition.name}(dimensions = {dimensions}, inertia = {inertia})"

    print(f"Running {name}")
    result = algorithm.run()
    print(f"Finished {name}: {result}")

    save_metrics(
        name=f"{name}: Runtime",
        metrics=algorithm.metrics_runtime,
        metric_names=("generation", "runtime"),
    )
    save_metrics(
        name=f"{name}: Fitness",
        metrics=algorithm.metrics_fitness,
        metric_names=("generation", "fitness"),
    )
    save_metrics(
        name=f"{name}: Values",
        metrics=algorithm.metrics_values,
        metric_names=("generation", "values"),
    )


if __name__ == "__main__":
    main()
