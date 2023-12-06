#!/usr/bin/env python3
import typing as t
import numpy as np
from itertools import chain
from multiprocessing import Process, Semaphore
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
    instances = wrap_instances_in_processes(instances, 8)

    instances()


def build_instances():
    return chain(
        build_discrete_instances(),
        build_stochastic_instances(),
    )


def build_discrete_instances():
    return iter(
        (
            function_definition,
            dimension,
            inertia_weight,
            cognitive_parameter,
            social_parameter,
            random_jitter_parameter,
            "outputs/discrete",
            MeteredParticleSwarmOptimisation.from_function_definition(
                function_definition=function_definition,
                dimensions=dimension,
                generations=100,
                inertia_weight=inertia_weight,
                cognitive_parameter=cognitive_parameter,
                social_parameter=social_parameter,
            ),
        )
        for inertia_weight in [0.0, 0.25, 0.5, 0.75, 1.0]
        for cognitive_parameter in [0.0, 0.25, 0.5, 0.75, 1.0]
        for social_parameter in [0.0, 0.25, 0.5, 0.75, 1.0]
        for random_jitter_parameter in [0.0, 0.25, 0.5, 1, 2]
        for dimension in [2, 30, 100]
        for function_definition in [
            rosenbrock_definition,
            michalewicz_definition,
            griewangk_definition,
            rastrigin_definition,
        ]
    )


def build_stochastic_instances():
    return iter(
        (
            function_definition,
            dimension,
            inertia_weight,
            cognitive_parameter,
            social_parameter,
            random_jitter_parameter,
            "outputs/stochastic",
            MeteredParticleSwarmOptimisation.from_function_definition(
                function_definition=function_definition,
                dimensions=dimension,
                generations=100,
                inertia_weight=inertia_weight,
                cognitive_parameter=cognitive_parameter,
                social_parameter=social_parameter,
            ),
        )
        for (
            inertia_weight,
            cognitive_parameter,
            social_parameter,
            random_jitter_parameter,
        ) in [
            (
                np.random.uniform(low=0.0, high=1.0),
                np.random.uniform(low=0.0, high=1.0),
                np.random.uniform(low=0.0, high=1.0),
                np.random.uniform(low=0.0, high=5.0),
            )
            for _ in range(100)
        ]
        for dimension in [2, 30, 100]
        for function_definition in [
            rosenbrock_definition,
            michalewicz_definition,
            griewangk_definition,
            rastrigin_definition,
        ]
    )


def wrap_instances_in_processes(
    instances: t.List[
        t.Tuple[FunctionDefinition, int, float, float, float, str, BaseAlgorithm]
    ],
    concurrency: int,
):
    def run():
        semaphore = Semaphore(concurrency)
        workers = []

        for (
            function_definition,
            dimensions,
            inertia,
            cognitive_parameter,
            team_bias,
            random_jitter_parameter,
            output_folder,
            algorithm,
        ) in instances:
            worker = Process(
                target=process,
                args=(
                    function_definition,
                    dimensions,
                    inertia,
                    cognitive_parameter,
                    team_bias,
                    random_jitter_parameter,
                    output_folder,
                    algorithm,
                    semaphore,
                ),
            )
            worker.start()
            workers.append(worker)
            semaphore.acquire()

        for worker in workers:
            worker.join()

    return run


def process(
    function_definition: FunctionDefinition,
    dimensions: int,
    inertia_weight: float,
    cognitive_parameter: float,
    social_parameter: float,
    random_jitter_parameter: float,
    output_folder: str,
    algorithm: BaseAlgorithm,
    semaphore: Semaphore,
):
    try:
        name = f"{algorithm.name}: {function_definition.name}(dimensions = {dimensions}, inertia = {inertia_weight}, team_bias = {social_parameter}, cognitive_parameter = {cognitive_parameter}, random_jitter_parameter = {random_jitter_parameter})"

        print(f"Running {name}")
        result = algorithm.run()
        print(f"Finished {name}: {result[1]} for {result[2]} generations")

        save_metrics(
            name=f"{name}: Runtime",
            metrics=algorithm.metrics_runtime,
            metric_names=("generation", "runtime"),
            output_folder=output_folder,
        )
        save_metrics(
            name=f"{name}: Fitness",
            metrics=algorithm.metrics_fitness,
            metric_names=("generation", "fitness"),
            output_folder=output_folder,
        )
        save_metrics(
            name=f"{name}: Values",
            metrics=algorithm.metrics_values,
            metric_names=("generation", "values"),
            output_folder=output_folder,
        )
    finally:
        semaphore.release()


if __name__ == "__main__":
    main()
