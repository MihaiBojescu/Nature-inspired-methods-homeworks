#!/usr/bin/env python3
import typing as t
from itertools import chain
from multiprocessing import Process, Semaphore
from algorithms.apso.metered_algorithm import MeteredAdaptiveParticleSwarmOptimisation
from algorithms.ga.metered_algorithm import MeteredGeneticAlgorithm
from algorithms.base.algorithm import BaseAlgorithm
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from functions.combinatorial.tsp.definitions.eil51 import make_eil51
from functions.combinatorial.tsp.definitions.berlin52 import make_berlin52
from functions.combinatorial.tsp.definitions.eil76 import make_eil76
from functions.combinatorial.tsp.definitions.rat99 import make_rat99
from functions.combinatorial.tsp.util.common import InitialPopulationGenerator
from functions.combinatorial.tsp.algorithms.apso.operators import PathRelinkingOperator, SwapOperator, TwoOptOperator
from functions.combinatorial.tsp.algorithms.ga.operators import (
    ComplexMutationOperator,
    CrossoverOperator,
    TournamentSelectionOperator,
)
from functions.combinatorial.tsp.util.encoder import Encoder
from util.sort import maximise, minimise
from util.import_export import save_metrics


def main():
    instances = build_instances()
    instances = wrap_instances_in_processes(instances, 7)

    instances()


def build_instances():
    return chain(
        build_discrete_instances(),
    )


def build_discrete_instances():
    return chain(
        iter(
            ga_tsp_generator(
                function_definition_constructor=function_definition_constructor,
                dimension=dimension,
                generations=generations,
                probabilities=probabilities,
            )
            for dimension in [2, 3]
            for function_definition_constructor in [make_eil51, make_berlin52, make_eil76, make_rat99]
            for generations in [5000, 10000]
            for probabilities in [
                [("mutation", 0.02)],
                [("mutation", 0.1)],
                [("mutation", 0.2)],
                [("mutation", 0.4)],
                [("mutation", 0.8)],
            ]
        ),
        iter(
            apso_tsp_generator(
                function_definition_constructor=function_definition_constructor,
                dimension=dimension,
                generations=generations,
                probabilities=probabilities,
            )
            for dimension in [2, 3, 5]
            for function_definition_constructor in [make_eil51, make_berlin52, make_eil76, make_rat99]
            for generations in [100, 500]
            for probabilities in [
                [("2-opt", 0.1), ("path-relink", 0.2), ("swap", 0.7)],
                [("2-opt", 0.3), ("path-relink", 0.2), ("swap", 0.5)],
                [("2-opt", 0.3333), ("path-relink", 0.3333), ("swap", 0.3334)],
                [("2-opt", 0.4), ("path-relink", 0.4), ("swap", 0.2)],
            ]
        )
    )


def ga_tsp_generator(
    function_definition_constructor: t.Callable[[int], CombinatorialFunctionDefinition],
    dimension: int,
    generations: int,
    probabilities: t.List[t.Tuple[str, float]],
):
    function_definition = function_definition_constructor(dimension)
    return (
        function_definition,
        dimension,
        generations,
        probabilities,
        "outputs/discrete",
        MeteredGeneticAlgorithm.from_function_definition(
            function_definition=function_definition,
            generate_initial_population=InitialPopulationGenerator(
                function_definition=function_definition, population_size=100
            ),
            criteria_function=lambda _best_individual, _best_individual_fitness, generation: generation > generations,
            selection_operator=TournamentSelectionOperator(
                fitness_compare_function=maximise if function_definition.target == "maximise" else minimise,
                tournament_size=10,
            ),
            crossover_operator=CrossoverOperator(
                encoder=Encoder(segments=function_definition.segmentation),
                fitness_function=function_definition.function,
            ),
            mutation_operator=ComplexMutationOperator(
                encoder=Encoder(segments=function_definition.segmentation), probability=probabilities[0][1]
            ),
        ),
    )


def apso_tsp_generator(
    function_definition_constructor: t.Callable[[int], CombinatorialFunctionDefinition],
    dimension: int,
    generations: int,
    probabilities: t.List[t.Tuple[str, float]],
):
    function_definition = function_definition_constructor(dimension)
    return (
        function_definition,
        dimension,
        generations,
        probabilities,
        "outputs/discrete",
        MeteredAdaptiveParticleSwarmOptimisation.from_function_definition(
            function_definition=function_definition,
            generate_initial_population=InitialPopulationGenerator(
                function_definition=function_definition, population_size=100
            ),
            criteria_function=lambda _best_individual, _best_individual_fitness, generation: generation > generations,
            two_opt_operator=TwoOptOperator(
                fitness_function=function_definition.function,
                fitness_compare_function=maximise if function_definition.target == "maximise" else minimise,
            ),
            path_linker_operator=PathRelinkingOperator(
                fitness_function=function_definition.function,
                fitness_compare_function=maximise if function_definition.target == "maximise" else minimise,
            ),
            swap_operator=SwapOperator(),
            two_opt_operator_probability=probabilities[0][1],
            path_linker_operator_probability=probabilities[1][1],
            swap_operator_probability=probabilities[2][1],
        ),
    )


def wrap_instances_in_processes(
    instances: t.List[
        t.Tuple[
            CombinatorialFunctionDefinition,
            int,
            int,
            t.List[t.Tuple[str, float]],
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
            probabilities,
            output_folder,
            algorithm,
        ) in instances:
            worker = Process(
                target=process,
                args=(
                    function_definition,
                    dimensions,
                    generations,
                    probabilities,
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
    probabilities: t.List[t.Tuple[str, float]],
    output_folder: str,
    algorithm: BaseAlgorithm,
    semaphore: Semaphore,
):
    try:
        hyperparameters_string = ", ".join([f"{name} = {value}" for name, value in probabilities])
        name = f"{algorithm.name}: {function_definition.name}(dimensions = {dimensions}, generations = {generations}, {hyperparameters_string})"

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
