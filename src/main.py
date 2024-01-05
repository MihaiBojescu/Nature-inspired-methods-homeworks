#!/usr/bin/env python3
import typing as t
from itertools import chain
from multiprocessing import Process, Semaphore
from algorithms.apso.metered_algorithm import MeteredAdaptiveParticleSwarmOptimisation
from algorithms.base.algorithm import BaseAlgorithm
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from functions.combinatorial.tsp.eil51 import make_eil51
from functions.combinatorial.tsp.common import InitialPopulationGenerator
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
        tsp_generator(
            function_definition_constructor=function_definition_constructor,
            dimension=dimension,
            generations=generations,
        )
        for dimension in [2]
        for function_definition_constructor in [
            make_eil51,
        ]
        for generations in [2000]
    )


def tsp_generator(
    function_definition_constructor: t.Callable[[int], CombinatorialFunctionDefinition],
    dimension: int,
    generations: int,
):
    function_definition = function_definition_constructor(dimension)
    return (
        function_definition,
        dimension,
        generations,
        "outputs/discrete",
        MeteredAdaptiveParticleSwarmOptimisation.from_function_definition(
            function_definition=function_definition,
            generate_initial_population=InitialPopulationGenerator(
                function_definition=function_definition, population_size=20
            ),
            dimensions=dimension,
        ),
    )


def wrap_instances_in_processes(
    instances: t.List[
        t.Tuple[
            CombinatorialFunctionDefinition,
            int,
            int,
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
            output_folder,
            algorithm,
        ) in instances:
            worker = Process(
                target=process,
                args=(
                    function_definition,
                    dimensions,
                    generations,
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
    output_folder: str,
    algorithm: BaseAlgorithm,
    semaphore: Semaphore,
):
    try:
        name = f"{algorithm.name}: {function_definition.name}(dimensions = {dimensions}, generations = {generations})"

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
