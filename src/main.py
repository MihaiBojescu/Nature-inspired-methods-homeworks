#!/usr/bin/env python3
from threading import Thread
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
    threads = [
        Thread(target=lambda: run_rastrigin(2)),
        # Thread(target=lambda: run_rastrigin(30)),
        # Thread(target=lambda: run_rastrigin(100)),
        Thread(target=lambda: run_griewangk(2)),
        # Thread(target=lambda: run_griewangk(30)),
        # Thread(target=lambda: run_griewangk(100)),
        Thread(target=lambda: run_rosenbrock_valley(2)),
        # Thread(target=lambda: run_rosenbrock_valley(30)),
        # Thread(target=lambda: run_rosenbrock_valley(100)),
        Thread(target=lambda: run_michalewicz(2)),
        # Thread(target=lambda: run_michalewicz(30)),
        # Thread(target=lambda: run_michalewicz(100)),
        Thread(target=lambda: run_h1p()),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # draw_continuous_hillclimber_best_score(2)
    # draw_continuous_hillclimber_runtime(2)
    # draw_binary_genetic_algorithm_fitness(2)
    # draw_binary_genetic_algorithm_runtime(2)
    # draw_hybrid_algorithm_fitness(2)
    # draw_hybrid_algorithm_runtime(2)


if __name__ == "__main__":
    main()
