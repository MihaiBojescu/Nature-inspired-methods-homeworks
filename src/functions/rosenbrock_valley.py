import math
import typing as t
import numpy as np
from metered_algorithms.metered_continuous_hillclimber import (
    MeteredContinuousHillclimber,
)
from metered_algorithms.metered_binary_genetic_algorithm import MeteredBinaryGenericAlgorithm
from metered_algorithms.metered_hybrid_algorithm import MeteredHybridAlgorithm
from util.import_export import save_metrics


module = "Rosenbrock valley"


def rosenbrock_valley(x: t.List[np.float32]):
    return np.sum(
        [100 * (x[i] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x))]
    )


def run_rosenbrock_valley(dimensions: int):
    run_hillclimber(dimensions)
    run_binary_genetic_algorithm(dimensions)
    run_hybrid_algorithm(dimensions)


def run_hillclimber(dimensions: int):
    hillclimber_algorithm = MeteredContinuousHillclimber(
        generate_initial_value=lambda: np.array(
            [np.float32(np.random.uniform(-2.048, 2.048)) for _ in range(dimensions)]
        ),
        fitness_function=rosenbrock_valley,
        fitness_compare_function=lambda a, b: a < b,
        neighbor_selection_function=None,
        criteria_function=lambda best_score, best_value, generations: generations
        >= 100,
        step=np.float32(0.1),
        acceleration=np.float32(0.1),
    )
    hillclimber_result = hillclimber_algorithm.run()

    print(
        f"{module}(dimensions = {dimensions}) - Continuous hillclimber results: {hillclimber_result}"
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Continuous hillclimber results: Runtime",
        hillclimber_algorithm.metrics_runtime,
        ("generation", "runtime"),
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Continuous hillclimber results: Best x",
        hillclimber_algorithm.metrics_best_value,
        ("generation", "x"),
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Continuous hillclimber results: Best step",
        hillclimber_algorithm.metrics_best_step,
        ("generation", "step"),
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Continuous hillclimber results: Best score",
        hillclimber_algorithm.metrics_best_score,
        ("generation", "score"),
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
            [np.float32(np.random.uniform(-2.048, 2.048)) for _ in range(dimensions)]
            for _ in range(dimensions)
        ],
        fitness_function=rosenbrock_valley,
        fitness_compare_function=lambda a, b: a < b,
        selection_function=selection_function,
        criteria_function=lambda generation, population: generation > 100,
        crossover_points=[np.uint32(4), np.uint32(9)],
        mutation_chance=np.float16(0.0001),
    )
    genetic_result = genetic_algorithm.run()

    print(
        f"{module}(dimensions = {dimensions}) - Binary genetic algorithm results: {genetic_result}"
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Binary genetic algorithm results: Runtime",
        genetic_algorithm.metrics_runtime,
        ("generation", "runtime"),
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Binary genetic algorithm results: Values",
        genetic_algorithm.metrics_values,
        ("generation", "value"),
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Binary genetic algorithm results: Fitness",
        genetic_algorithm.metrics_fitness,
        ("generation", "fitness"),
    )


def run_hybrid_algorithm(dimensions: int):
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
            [np.float32(np.random.uniform(-2.048, 2.048)) for _ in range(dimensions)]
            for _ in range(dimensions)
        ],
        fitness_function=rosenbrock_valley,
        fitness_compare_function=lambda a, b: a < b,
        selection_function=selection_function,
        criteria_function=lambda generation, population: generation > 100,
        crossover_points=[np.uint32(4), np.uint32(9)],
        mutation_chance=np.float16(0.0001),
        hillclimber_neighbor_selection_function=None,
        hillclimber_run_interval=10,
    )
    hybrid_result = hybrid_algorithm.run()

    print(
        f"{module}(dimensions = {dimensions}) - Hybrid algorithm results: {hybrid_result}"
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Hybrid algorithm results: Runtime",
        hybrid_algorithm.metrics_runtime,
        ("generation", "runtime"),
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Hybrid algorithm results: Values",
        hybrid_algorithm.metrics_values,
        ("generation", "value"),
    )
    save_metrics(
        f"{module}(dimensions = {dimensions}) - Hybrid algorithm results: Fitness",
        hybrid_algorithm.metrics_fitness,
        ("generation", "fitness"),
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
