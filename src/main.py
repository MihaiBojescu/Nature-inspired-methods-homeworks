#!/usr/bin/env python3

import math
import random
import numpy as np
from metered_algorithms.metered_continuous_hillclimber import (
    MeteredContinuousHillclimber,
)
from metered_algorithms.metered_genetic_algorithm import MeteredBinaryGenericAlgorithm
from metered_algorithms.metered_hybrid_algorithm import MeteredHybridAlgorithm
from util.graph import graph_genetic_algorithm, graph_hillclimber


def main():
    hillclimber_algorithm = MeteredContinuousHillclimber(
        fx=lambda x: x**3 - 60 * (x**2) + 900 * x + 100,
        initial_x=(np.float32(0.0), np.float32(31.0)),
        step=np.float32(0.1),
        acceleration=np.float32(0.1),
        precision=np.finfo(np.float32).eps,
        generations=100,
    )
    hillclimber_result = hillclimber_algorithm.run()

    print(hillclimber_result)
    graph_hillclimber(
        "Hillclimber results: Best X", "X", hillclimber_algorithm.metrics_best_x
    )
    graph_hillclimber(
        "Hillclimber results: Best step",
        "step",
        hillclimber_algorithm.metrics_best_step,
    )
    graph_hillclimber(
        "Hillclimber results: Best score",
        "score",
        hillclimber_algorithm.metrics_best_score,
    )

    genetic_algorithm = MeteredBinaryGenericAlgorithm(
        encode=lambda x: np.frombuffer(
            np.array([x], dtype=np.float32).tobytes(), dtype=np.uint8
        ),
        decode=lambda x: np.frombuffer(x.tobytes(), dtype=np.float32)[0],
        generate_initial_population=lambda: [
            np.float32(random.random() * 31) for _ in range(0, 100)
        ],
        fitness_function=lambda x: x**3 - 60 * (x**2) + 900 * x + 100,
        criteria_function=lambda generation, population: generation > 100,
        selection_function=selection_function,
        crossover_points=[np.uint32(4), np.uint32(9)],
        mutation_chance=np.float16(0.0001),
    )
    genetic_result = genetic_algorithm.run()

    print(genetic_result)
    graph_genetic_algorithm(
        "Genetic algorithm: values",
        genetic_algorithm.metrics_values,
    )
    graph_genetic_algorithm(
        "Genetic algorithm: Fitness",
        genetic_algorithm.metrics_fitness,
    )

    hybrid_algorithm = MeteredHybridAlgorithm(
        encode=lambda x: np.frombuffer(
            np.array([x], dtype=np.float32).tobytes(), dtype=np.uint8
        ),
        decode=lambda x: np.frombuffer(x.tobytes(), dtype=np.float32)[0],
        generate_initial_population=lambda: [
            np.float32(random.random() * 31) for _ in range(0, 100)
        ],
        fitness_function=lambda x: x**3 - 60 * (x**2) + 900 * x + 100,
        criteria_function=lambda generation, population: generation > 100,
        selection_function=selection_function,
        crossover_points=[np.uint32(4), np.uint32(9)],
        mutation_chance=np.float16(0.0001),
        hillclimber_run_interval=10,
    )
    hybrid_result = hybrid_algorithm.run()

    print(hybrid_result)
    graph_genetic_algorithm(
        "Hybrid algorithm: Values",
        hybrid_algorithm.metrics_values,
    )
    graph_genetic_algorithm(
        "Hybrid algorithm: Fitness",
        hybrid_algorithm.metrics_fitness,
    )


def selection_function(population):
    parent_1 = population[0]
    parent_2 = population[0]

    for individual in population:
        if (
            0 < individual[0] < 31
            and not math.isnan(individual[1])
            and not math.isinf(individual[1])
        ):
            parent_1 = individual
            break

    for individual in population:
        if (
            0 < individual[0] < 31
            and not math.isnan(individual[1])
            and not math.isinf(individual[1])
            and individual != parent_1
        ):
            parent_2 = individual
            break

    return parent_1, parent_2


if __name__ == "__main__":
    main()
