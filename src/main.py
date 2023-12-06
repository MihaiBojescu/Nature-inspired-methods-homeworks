#!/usr/bin/env python3
import typing as t
from multiprocessing import Process, Semaphore
from itertools import chain
from functions.griewangk import griewangk_definition
from functions.h1p import h1p_definition
from functions.michalewicz import michalewicz_definition
from functions.rastrigin import rastrigin_definition
from functions.rosenbrock import rosenbrock_definition
from functions.definition import FunctionDefinition
from algorithms.base.algorithm import BaseAlgorithm
from algorithms.binary_hillclimber.metered_binary_hillclimber import MeteredBinaryHillclimber
from algorithms.continuous_hillclimber.metered_continuous_hillclimber import MeteredContinuousHillclimber
from algorithms.binary_genetic_algorithm.metered_binary_genetic_algorithm import MeteredBinaryGenericAlgorithm
from algorithms.hybrid_algorithm.metered_hybrid_algorithm import MeteredHybridAlgorithm
from util.import_export import save_metrics
from util.graph import (
    draw_binary_genetic_algorithm_fitness,
    draw_binary_genetic_algorithm_runtime,
    draw_continuous_hillclimber_best_score,
    draw_continuous_hillclimber_runtime,
    draw_hybrid_algorithm_fitness,
    draw_hybrid_algorithm_runtime,
)


def main():
    instances = build_instances()
    instances = wrap_instances_in_processes(instances, 8)

    instances()

    # for dimensions in [2, 30, 100]:
    #     draw_continuous_hillclimber_best_score(dimensions)
    #     draw_continuous_hillclimber_runtime(dimensions)
    #     draw_binary_genetic_algorithm_fitness(dimensions)
    #     draw_binary_genetic_algorithm_runtime(dimensions)
    #     draw_hybrid_algorithm_fitness(dimensions)
    #     draw_hybrid_algorithm_runtime(dimensions)


def build_instances():
    return chain(
        iter(
            (
                function_definition,
                dimension,
                MeteredBinaryHillclimber.from_function_definition(
                    function_definition=function_definition,
                    dimensions=dimension,
                    generations=100,
                ),
            )
            for dimension in [2, 30, 100]
            for function_definition in [
                rosenbrock_definition,
                michalewicz_definition,
                griewangk_definition,
                rastrigin_definition,
            ]
        ),
        iter(
            (
                function_definition,
                dimension,
                MeteredContinuousHillclimber.from_function_definition(
                    function_definition=function_definition,
                    dimensions=dimension,
                    generations=100,
                ),
            )
            for dimension in [2, 30, 100]
            for function_definition in [
                rosenbrock_definition,
                michalewicz_definition,
                griewangk_definition,
                rastrigin_definition,
            ]
        ),
        iter(
            (
                function_definition,
                dimension,
                MeteredBinaryGenericAlgorithm.from_function_definition(
                    function_definition=function_definition,
                    dimensions=dimension,
                    generations=100,
                ),
            )
            for dimension in [2, 30, 100]
            for function_definition in [
                rosenbrock_definition,
                michalewicz_definition,
                griewangk_definition,
                rastrigin_definition,
            ]
        ),
        iter(
            (
                function_definition,
                dimension,
                MeteredHybridAlgorithm.from_function_definition(
                    function_definition=function_definition,
                    dimensions=dimension,
                    generations=100,
                ),
            )
            for dimension in [2, 30, 100]
            for function_definition in [
                rosenbrock_definition,
                michalewicz_definition,
                griewangk_definition,
                rastrigin_definition,
            ]
        ),
    )

def wrap_instances_in_processes(
    instances: t.List[
        t.Tuple[FunctionDefinition, int, BaseAlgorithm]
    ],
    concurrency: int,
):
    def run():
        semaphore = Semaphore(concurrency)
        workers = []

        for (
            function_definition,
            dimensions,
            algorithm,
        ) in instances:
            worker = Process(
                target=process,
                args=(
                    function_definition,
                    dimensions,
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
    algorithm: BaseAlgorithm,
    semaphore: Semaphore,
):
    name = f"{algorithm.name}: {function_definition.name}(dimensions = {dimensions})"

    print(f"Running {name}")
    result = algorithm.run()
    print(f"Finished {name}: {result[1]} for {result[2]} generations")

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
    semaphore.release()

if __name__ == "__main__":
    main()
