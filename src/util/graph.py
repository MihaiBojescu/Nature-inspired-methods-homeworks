import os
import typing as t
import numpy as np
import matplotlib.pyplot as plt
from util.import_export import load_metrics


def draw_continuous_hillclimber_best_score(dimension: int):
    rastringin_best_score = load_metrics(
        f"Rastrigin(dimensions = {dimension}) - Continuous hillclimber results: Best score"
    )
    griewangk_best_score = load_metrics(
        f"Griewangk(dimensions = {dimension}) - Continuous hillclimber results: Best score"
    )
    rosenbrock_best_score = load_metrics(
        f"Rosenbrock valley(dimensions = {dimension}) - Continuous hillclimber results: Best score"
    )
    michalewicz_best_score = load_metrics(
        f"Michalewicz(dimensions = {dimension}) - Continuous hillclimber results: Best score"
    )

    data = [
        ("Rastringin's", rastringin_best_score),
        ("Griewangk's", griewangk_best_score),
        ("Rosenbrock's valley", rosenbrock_best_score),
        ("Michalewicz's", michalewicz_best_score),
    ]
    data = list(
        map(
            lambda entry: (
                entry[0],
                [metric[0] for metric in entry[1]],
                [float(metric[1]) for metric in entry[1]],
            ),
            data,
        )
    )

    os.makedirs("../images", exist_ok=True)

    for metrics in data:
        name = metrics[0]
        plt.plot(metrics[1], metrics[2], label=name)

    plt.legend(loc="best")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.yscale("log")
    plt.savefig(
        f"../images/hillclimber best score(dimension = {dimension}).svg",
        format="svg",
    )
    plt.show()


def draw_continuous_hillclimber_runtime(dimension: int):
    rastringin_runtime = load_metrics(
        f"Rastrigin(dimensions = {dimension}) - Continuous hillclimber results: Runtime"
    )
    griewangk_runtime = load_metrics(
        f"Griewangk(dimensions = {dimension}) - Continuous hillclimber results: Runtime"
    )
    rosenbrock_runtime = load_metrics(
        f"Rosenbrock valley(dimensions = {dimension}) - Continuous hillclimber results: Runtime"
    )
    michalewicz_runtime = load_metrics(
        f"Michalewicz(dimensions = {dimension}) - Continuous hillclimber results: Runtime"
    )

    data = [
        ("Rastringin's", rastringin_runtime),
        ("Griewangk's", griewangk_runtime),
        ("Rosenbrock's valley", rosenbrock_runtime),
        ("Michalewicz's", michalewicz_runtime),
    ]
    data = list(
        map(
            lambda entry: (
                entry[0],
                [metric[0] for metric in entry[1]],
                [
                    (float(metric[1]) - min([float(metric[1]) for metric in entry[1]]))
                    / 1000000000
                    for metric in entry[1]
                ],
            ),
            data,
        )
    )

    os.makedirs("../images", exist_ok=True)

    for metrics in data:
        name = metrics[0]
        plt.plot(metrics[2], metrics[1], label=name)

    plt.legend(loc="best")
    plt.xlabel("Time in seconds")
    plt.ylabel("Generation")
    plt.savefig(
        f"../images/binary genetic algorithm runtimes(dimension = {dimension}).svg",
        format="svg",
    )
    plt.show()


def draw_binary_genetic_algorithm_fitness(dimension: int):
    rastringin_fitness = load_metrics(
        f"Rastrigin(dimensions = {dimension}) - Binary genetic algorithm results: Fitness"
    )
    griewangk_fitness = load_metrics(
        f"Griewangk(dimensions = {dimension}) - Binary genetic algorithm results: Fitness"
    )
    rosenbrock_fitness = load_metrics(
        f"Rosenbrock valley(dimensions = {dimension}) - Binary genetic algorithm results: Fitness"
    )
    michalewicz_fitness = load_metrics(
        f"Michalewicz(dimensions = {dimension}) - Binary genetic algorithm results: Fitness"
    )

    data = [
        ("Rastringin's", rastringin_fitness),
        ("Griewangk's", griewangk_fitness),
        ("Rosenbrock's valley", rosenbrock_fitness),
        ("Michalewicz's", michalewicz_fitness),
    ]
    data = list(
        map(
            lambda entry: (
                entry[0],
                [metric[0] for metric in entry[1]],
                [
                    (float(metric[1]) - min([float(metric[1]) for metric in entry[1]]))
                    / 1000000000
                    for metric in entry[1]
                ],
            ),
            data,
        )
    )

    os.makedirs("../images", exist_ok=True)

    for metrics in data:
        name = metrics[0]
        plt.plot(metrics[1], metrics[2], label=name)

    plt.legend(loc="best")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(
        f"../images/binary genetic algorithm fitness(dimension = {dimension}).svg",
        format="svg",
    )
    plt.show()


def draw_binary_genetic_algorithm_runtime(dimension: int):
    rastringin_best_score = load_metrics(
        f"Rastrigin(dimensions = {dimension}) - Binary genetic algorithm results: Runtime"
    )
    griewangk_best_score = load_metrics(
        f"Griewangk(dimensions = {dimension}) - Binary genetic algorithm results: Runtime"
    )
    rosenbrock_best_score = load_metrics(
        f"Rosenbrock valley(dimensions = {dimension}) - Binary genetic algorithm results: Runtime"
    )
    michalewicz_best_score = load_metrics(
        f"Michalewicz(dimensions = {dimension}) - Binary genetic algorithm results: Runtime"
    )

    data = [
        ("Rastringin's", rastringin_best_score),
        ("Griewangk's", griewangk_best_score),
        ("Rosenbrock's valley", rosenbrock_best_score),
        ("Michalewicz's", michalewicz_best_score),
    ]
    data = list(
        map(
            lambda entry: (
                entry[0],
                [metric[0] for metric in entry[1]],
                [
                    (float(metric[1]) - min([float(metric[1]) for metric in entry[1]]))
                    / 1000000000
                    for metric in entry[1]
                ],
            ),
            data,
        )
    )

    os.makedirs("../images", exist_ok=True)

    for metrics in data:
        name = metrics[0]
        plt.plot(metrics[2], metrics[1], label=name)

    plt.legend(loc="best")
    plt.xlabel("Time in seconds")
    plt.ylabel("Generation")
    plt.savefig(
        f"../images/binary genetic algorithm runtimes(dimension = {dimension}).svg",
        format="svg",
    )
    plt.show()


def draw_hybrid_algorithm_fitness(dimension: int):
    rastringin_fitness = load_metrics(
        f"Rastrigin(dimensions = {dimension}) - Hybrid algorithm results: Fitness"
    )
    griewangk_fitness = load_metrics(
        f"Griewangk(dimensions = {dimension}) - Hybrid algorithm results: Fitness"
    )
    rosenbrock_fitness = load_metrics(
        f"Rosenbrock valley(dimensions = {dimension}) - Hybrid algorithm results: Fitness"
    )
    michalewicz_fitness = load_metrics(
        f"Michalewicz(dimensions = {dimension}) - Hybrid algorithm results: Fitness"
    )

    data = [
        ("Rastringin's", rastringin_fitness),
        ("Griewangk's", griewangk_fitness),
        ("Rosenbrock's valley", rosenbrock_fitness),
        ("Michalewicz's", michalewicz_fitness),
    ]
    data = list(
        map(
            lambda entry: (
                entry[0],
                [metric[0] for metric in entry[1]],
                [
                    (float(metric[1]) - min([float(metric[1]) for metric in entry[1]]))
                    / 1000000000
                    for metric in entry[1]
                ],
            ),
            data,
        )
    )

    os.makedirs("../images", exist_ok=True)

    for metrics in data:
        name = metrics[0]
        plt.plot(metrics[1], metrics[2], label=name)

    plt.legend(loc="best")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(
        f"../images/hybrid genetic algorithm fitness(dimension = {dimension}).svg",
        format="svg",
    )
    plt.show()


def draw_hybrid_algorithm_runtime(dimension: int):
    rastringin_best_score = load_metrics(
        f"Rastrigin(dimensions = {dimension}) - Hybrid algorithm results: Runtime"
    )
    griewangk_best_score = load_metrics(
        f"Griewangk(dimensions = {dimension}) - Hybrid algorithm results: Runtime"
    )
    rosenbrock_best_score = load_metrics(
        f"Rosenbrock valley(dimensions = {dimension}) - Hybrid algorithm results: Runtime"
    )
    michalewicz_best_score = load_metrics(
        f"Michalewicz(dimensions = {dimension}) - Hybrid algorithm results: Runtime"
    )

    data = [
        ("Rastringin's", rastringin_best_score),
        ("Griewangk's", griewangk_best_score),
        ("Rosenbrock's valley", rosenbrock_best_score),
        ("Michalewicz's", michalewicz_best_score),
    ]
    data = list(
        map(
            lambda entry: (
                entry[0],
                [metric[0] for metric in entry[1]],
                [
                    (float(metric[1]) - min([float(metric[1]) for metric in entry[1]]))
                    / 1000000000
                    for metric in entry[1]
                ],
            ),
            data,
        )
    )

    os.makedirs("../images", exist_ok=True)

    for metrics in data:
        name = metrics[0]
        plt.plot(metrics[2], metrics[1], label=name)

    plt.legend(loc="best")
    plt.xlabel("Time in seconds")
    plt.ylabel("Generation")
    plt.savefig(
        f"../images/hybrid runtimes(dimension = {dimension}).svg",
        format="svg",
    )
    plt.show()
