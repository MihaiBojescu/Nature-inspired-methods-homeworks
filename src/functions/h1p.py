import math
import random
import numpy as np
from metered_algorithms.metered_continuous_hillclimber import (
    MeteredContinuousHillclimber,
)
from metered_algorithms.metered_binary_genetic_algorithm import (
    MeteredBinaryGenericAlgorithm,
)
from metered_algorithms.metered_hybrid_algorithm import MeteredHybridAlgorithm
from util.import_export import save_metrics

module = "H1'"


def h1p(x: np.uint32):
    return x**3 - 60 * (x**2) + 900 * x + 100


def run_h1p(population_size: int):
    run_hillclimber()
    run_binary_genetic_algorithm(population_size)
    run_hybrid_algorithm(population_size)


def run_hillclimber():
    hillclimber_algorithm = MeteredContinuousHillclimber(
        generate_initial_value=lambda: np.random.uniform(
            low=np.float32(0.0), high=np.float32(31.0)
        ),
        fitness_function=h1p,
        fitness_compare_function=lambda a, b: a > b,
        neighbor_selection_function=lambda value: 0 < value < 31,
        criteria_function=lambda best_score, best_value, generations: generations
        >= 100,
        step=np.float32(0.1),
        acceleration=np.float32(0.1),
    )
    hillclimber_result = hillclimber_algorithm.run()

    print(f"{module} - Continuous hillclimber results: {hillclimber_result}")
    save_metrics(
        f"{module} - Continuous hillclimber results: Runtime",
        hillclimber_algorithm.metrics_runtime,
        ("generation", "runtime"),
    )
    save_metrics(
        f"{module} - Continuous hillclimber results: Best value",
        hillclimber_algorithm.metrics_best_value,
        ("generation", "x"),
    )
    save_metrics(
        f"{module} - Continuous hillclimber results: Best step",
        hillclimber_algorithm.metrics_best_step,
        ("generation", "step"),
    )
    save_metrics(
        f"{module} - Continuous hillclimber results: Best score",
        hillclimber_algorithm.metrics_best_score,
        ("generation", "score"),
    )


def run_binary_genetic_algorithm(population_size: int):
    genetic_algorithm = MeteredBinaryGenericAlgorithm(
        encode=lambda x: np.frombuffer(
            np.array([x], dtype=np.float32).tobytes(), dtype=np.uint8
        ),
        decode=lambda x: np.frombuffer(x.tobytes(), dtype=np.float32)[0],
        generate_initial_population=lambda: [
            np.float32(random.random() * 31) for _ in range(population_size)
        ],
        fitness_function=h1p,
        fitness_compare_function=lambda a, b: a > b,
        selection_function=selection_function,
        criteria_function=lambda best_fitness, best_value, generation: generation
        >= 100,
        crossover_points=[np.uint32(4), np.uint32(9)],
        mutation_chance=np.float16(0.0001),
    )
    genetic_result = genetic_algorithm.run()

    print(f"{module} - Binary genetic algorithm results: {genetic_result}")
    save_metrics(
        f"{module} - Binary genetic algorithm results: Runtime",
        genetic_algorithm.metrics_runtime,
        ("generation", "runtime"),
    )
    save_metrics(
        f"{module} - Binary genetic algorithm results: Values",
        genetic_algorithm.metrics_values,
        ("generation", "value"),
    )
    save_metrics(
        f"{module} - Binary genetic algorithm results: Fitness",
        genetic_algorithm.metrics_fitness,
        ("generation", "fitness"),
    )


def run_hybrid_algorithm(population_size: int):
    hybrid_algorithm = MeteredHybridAlgorithm(
        encode=lambda x: np.frombuffer(
            np.array([x], dtype=np.float32).tobytes(), dtype=np.uint8
        ),
        decode=lambda x: np.frombuffer(x.tobytes(), dtype=np.float32)[0],
        generate_initial_population=lambda: [
            np.float32(random.random() * 31) for _ in range(population_size)
        ],
        fitness_function=h1p,
        fitness_compare_function=lambda a, b: a > b,
        selection_function=selection_function,
        criteria_function=lambda best_fitness, best_value, generation: generation
        >= 100,
        crossover_points=[np.uint32(4), np.uint32(9)],
        mutation_chance=np.float16(0.0001),
        hillclimber_neighbor_selection_function=None,
        hillclimber_run_interval=10,
    )
    hybrid_result = hybrid_algorithm.run()

    print(f"{module} - Hybrid algorithm results: {hybrid_result}")
    save_metrics(
        f"{module} - Hybrid algorithm results: Runtime",
        hybrid_algorithm.metrics_runtime,
        ("generation", "runtime"),
    )
    save_metrics(
        f"{module} - Hybrid algorithm results: Values",
        hybrid_algorithm.metrics_values,
        ("generation", "value"),
    )
    save_metrics(
        f"{module} - Hybrid algorithm results: Fitness",
        hybrid_algorithm.metrics_fitness,
        ("generation", "fitness"),
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
