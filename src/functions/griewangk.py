import math
import typing as t
import numpy as np
from metered_algorithms.metered_continuous_hillclimber import (
    MeteredContinuousHillclimber,
)
from metered_algorithms.metered_genetic_algorithm import MeteredBinaryGenericAlgorithm
from metered_algorithms.metered_hybrid_algorithm import MeteredHybridAlgorithm
from util.graph import graph_genetic_algorithm, graph_hillclimber

module = "Griewangk"


def griewangk(x: t.List[np.float32]):
    return (
        np.sum([x[i] ** 2 / 4000 for i in range(len(x))])
        - np.prod([np.cos(x[i] / np.sqrt[i]) for i in range(len(x))])
        + 1
    )


def run_griewangk(dimensions: int):
    run_hillclimber(dimensions)
    run_binary_genetic_algorithm(dimensions)
    run_hybric_algorithm(dimensions)


def run_hillclimber(dimensions: int):
    hillclimber_algorithm = MeteredContinuousHillclimber(
        fx=griewangk,
        initial_x=np.array(
            [np.float32(np.random.uniform(-100, 100)) for _ in range(dimensions)]
        ),
        step=np.float32(0.1),
        acceleration=np.float32(0.1),
        precision=np.finfo(np.float32).eps,
        generations=100,
    )
    hillclimber_result = hillclimber_algorithm.run()

    print(f"{module}(dimensions = {dimensions}) - Continuous hillclimber results: {hillclimber_result}")
    graph_hillclimber(
        f"{module}(dimensions = {dimensions}) - Hillclimber results: Best X", "X", hillclimber_algorithm.metrics_best_x
    )
    graph_hillclimber(
        f"{module}(dimensions = {dimensions}) - Hillclimber results: Best step",
        "step",
        hillclimber_algorithm.metrics_best_step,
    )
    graph_hillclimber(
        f"{module}(dimensions = {dimensions}) - Hillclimber results: Best score",
        "score",
        hillclimber_algorithm.metrics_best_score,
    )


def run_binary_genetic_algorithm(dimensions: int):
    genetic_algorithm = MeteredBinaryGenericAlgorithm(
        encode=lambda x: np.array(
            [
                value
                for xi in x
                for value in np.frombuffer(
                    np.array([xi], dtype=np.float32).tobytes(), dtype=np.uint8
                )
            ]
        ),
        decode=lambda x: [
            np.frombuffer(np.array(batch).tobytes(), dtype=np.float32)[0]
            for batch in [x[i : i + 4] for i in range(0, len(x), 4)]
        ],
        generate_initial_population=lambda: [
            [np.float32(np.random.uniform(-100, 100)) for _ in range(dimensions)]
            for _ in range(dimensions)
        ],
        fitness_function=griewangk,
        criteria_function=lambda generation, population: generation > 100,
        selection_function=selection_function,
        crossover_points=[np.uint32(4), np.uint32(9)],
        mutation_chance=np.float16(0.0001),
    )
    genetic_result = genetic_algorithm.run()

    print(f"{module}(dimensions = {dimensions}) - Binary genetic algorithm results: {genetic_result}")
    graph_genetic_algorithm(
        f"{module}(dimensions = {dimensions}) - Genetic algorithm: values",
        genetic_algorithm.metrics_values,
    )
    graph_genetic_algorithm(
        f"{module}(dimensions = {dimensions}) - Genetic algorithm: Fitness",
        genetic_algorithm.metrics_fitness,
    )


def run_hybric_algorithm(dimensions: int):
    hybrid_algorithm = MeteredHybridAlgorithm(
        encode=lambda x: np.array(
            [
                value
                for xi in x
                for value in np.frombuffer(
                    np.array([xi], dtype=np.float32).tobytes(), dtype=np.uint8
                )
            ]
        ),
        decode=lambda x: [
            np.frombuffer(np.array(batch).tobytes(), dtype=np.float32)[0]
            for batch in [x[i : i + 4] for i in range(0, len(x), 4)]
        ],
        generate_initial_population=lambda: [
            [np.float32(np.random.uniform(-100, 100)) for _ in range(dimensions)]
            for _ in range(dimensions)
        ],
        fitness_function=griewangk,
        criteria_function=lambda generation, population: generation > 100,
        selection_function=selection_function,
        crossover_points=[np.uint32(4), np.uint32(9)],
        mutation_chance=np.float16(0.0001),
        hillclimber_run_interval=10,
    )
    hybrid_result = hybrid_algorithm.run()

    print(f"{module}(dimensions = {dimensions}) - Hybrid algorithm results: {hybrid_result}")
    graph_genetic_algorithm(
        f"{module}(dimensions = {dimensions}) - Hybrid algorithm: Values",
        hybrid_algorithm.metrics_values,
    )
    graph_genetic_algorithm(
        f"{module}(dimensions = {dimensions}) - Hybrid algorithm: Fitness",
        hybrid_algorithm.metrics_fitness,
    )


def selection_function(population):
    parent_1 = population[0]
    parent_2 = population[0]

    for individual in population:
        if not math.isnan(individual[1]) and not math.isinf(individual[1]):
            parent_1 = individual
            break

    for individual in population:
        if (
            not math.isnan(individual[1])
            and not math.isinf(individual[1])
            and individual != parent_1
        ):
            parent_2 = individual
            break

    return parent_1, parent_2
