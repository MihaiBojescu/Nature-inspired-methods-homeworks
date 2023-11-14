#!/usr/bin/env python3
from multiprocessing import Process
from functions.griewangk import run_griewangk
from functions.h1p import run_h1p
from functions.michalewicz import run_michalewicz
from functions.rastringin import run_rastrigin
from functions.rosenbrock_valley import run_rosenbrock_valley
from algorithms.binary_hillclimber import BinaryHillclimber
from util.graph import (
    draw_binary_genetic_algorithm_fitness,
    draw_binary_genetic_algorithm_runtime,
    draw_continuous_hillclimber_best_score,
    draw_continuous_hillclimber_runtime,
    draw_hybrid_algorithm_fitness,
    draw_hybrid_algorithm_runtime,
)


def main():
    processes = [
        Process(target=lambda: run_rastrigin(population_size=100, dimensions=2)),
        Process(target=lambda: run_rastrigin(population_size=100, dimensions=30)),
        Process(target=lambda: run_rastrigin(population_size=100, dimensions=100)),
        Process(target=lambda: run_griewangk(population_size=100, dimensions=2)),
        Process(target=lambda: run_griewangk(population_size=100, dimensions=30)),
        Process(target=lambda: run_griewangk(population_size=100, dimensions=100)),
        Process(target=lambda: run_rosenbrock_valley(population_size=100, dimensions=2)),
        Process(target=lambda: run_rosenbrock_valley(population_size=100, dimensions=30)),
        Process(target=lambda: run_rosenbrock_valley(population_size=100, dimensions=100)),
        Process(target=lambda: run_michalewicz(population_size=100, dimensions=2)),
        Process(target=lambda: run_michalewicz(population_size=100, dimensions=30)),
        Process(target=lambda: run_michalewicz(population_size=100, dimensions=100)),
        Process(target=lambda: run_h1p(population_size=100)),
    ]

    for thread in processes:
        thread.start()

    for thread in processes:
        thread.join()

    for dimensions in [2, 30, 100]:
        draw_continuous_hillclimber_best_score(dimensions)
        draw_continuous_hillclimber_runtime(dimensions)
        draw_binary_genetic_algorithm_fitness(dimensions)
        draw_binary_genetic_algorithm_runtime(dimensions)
        draw_hybrid_algorithm_fitness(dimensions)
        draw_hybrid_algorithm_runtime(dimensions)


if __name__ == "__main__":
    main()
