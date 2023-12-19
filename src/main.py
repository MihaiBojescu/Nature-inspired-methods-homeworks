#!/usr/bin/env python3
import typing as t
from itertools import chain
from multiprocessing import Process, Semaphore
from algorithms.cpso.metered_algorithm import (
    MeteredCombinatorialParticleSwarmOptimisation,
)
from algorithms.base.algorithm import BaseAlgorithm
from functions.combinatorial.tsp.eil51 import eil51
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from functions.combinatorial.tsp.common import generate_initial_population
from util.import_export import save_metrics


def main():
    instances = build_instances()
    instances = wrap_instances_in_processes(instances, 8)

    instances()


def build_instances():
    return chain(
        build_discrete_instances(),
    )


def build_discrete_instances():
    return iter(
        (
            function_definition,
            dimension,
            generations,
            inertia_weight,
            cognitive_parameter,
            social_parameter,
            intensification_parameter,
            "outputs/discrete",
            MeteredCombinatorialParticleSwarmOptimisation.from_function_definition(
                function_definition=function_definition,
                generate_initial_population=generate_initial_population(
                    function_definition=function_definition,
                    population_size=100,
                    dimensions=dimension,
                ),
                dimensions=dimension,
                inertia_weight=inertia_weight,
                cognitive_parameter=cognitive_parameter,
                social_parameter=social_parameter,
            ),
        )
        for dimension in [2, 3]
        for function_definition in [
            eil51,
        ]
        for generations in [100]
        for inertia_weight in [0.0, 0.25, 0.5, 0.75, 1.0]
        for cognitive_parameter in [0.0, 0.25, 0.5, 0.75, 1.0]
        for social_parameter in [0.0, 0.25, 0.5, 0.75, 1.0]
        for intensification_parameter in [0.0, 0.5, 1.0]
    )


def wrap_instances_in_processes(
    instances: t.List[
        t.Tuple[
            CombinatorialFunctionDefinition,
            int,
            int,
            float,
            float,
            float,
            str,
            BaseAlgorithm,
        ]
    ],
    concurrency: int,
):
    def run():
        semaphore = Semaphore(concurrency)
        workers = []

        for (
            function_definition,
            dimensions,
            generations,
            inertia,
            cognitive_parameter,
            social_parameter,
            intensification_parameter,
            output_folder,
            algorithm,
        ) in instances:
            worker = Process(
                target=process,
                args=(
                    function_definition,
                    dimensions,
                    generations,
                    inertia,
                    cognitive_parameter,
                    social_parameter,
                    intensification_parameter,
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
    function_definition: CombinatorialFunctionDefinition,
    dimensions: int,
    generations: int,
    inertia_weight: float,
    cognitive_parameter: float,
    social_parameter: float,
    intensification_parameter: float,
    output_folder: str,
    algorithm: BaseAlgorithm,
    semaphore: Semaphore,
):
    try:
        name = f"{algorithm.name}: {function_definition.name}(dimensions = {dimensions}, generations = {generations}, inertia = {inertia_weight}, team_bias = {social_parameter}, cognitive_parameter = {cognitive_parameter}, intensification_parameter = {intensification_parameter})"

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
