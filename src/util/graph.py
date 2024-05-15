import os
import matplotlib.pyplot as plt
from util.import_export import load_metrics


def draw_fitness(dimension: int):
    exports_path = os.path.join(os.path.dirname(__file__), "../../images")

    rastringin_fitness = load_metrics(
        f"PSO algorithm: Rastrigin(dimensions = {dimension}, inertia = 0.2): Fitness - run 0"
    )
    griewangk_fitness = load_metrics(
        f"PSO algorithm: Griewangk(dimensions = {dimension}, inertia = 0.2): Fitness - run 0"
    )
    rosenbrock_fitness = load_metrics(
        f"PSO algorithm: Rosenbrock(dimensions = {dimension}, inertia = 0.2): Fitness - run 0"
    )
    michalewicz_fitness = load_metrics(
        f"PSO algorithm: Michalewicz(dimensions = {dimension}, inertia = 0.2): Fitness - run 0"
    )

    data = [
        ("Rastringin's", rastringin_fitness),
        ("Griewangk's", griewangk_fitness),
        ("Rosenbrock's", rosenbrock_fitness),
        ("Michalewicz's", michalewicz_fitness),
    ]

    data = list(
        map(
            lambda entry: (
                entry[0],
                [int(float(metric[0])) for metric in entry[1]],
                [float(metric[1]) for metric in entry[1]],
            ),
            data,
        )
    )
    visited = []
    results = []
    for entry in data:
        if (entry[0], entry[1]) in visited:
            continue

        visited.append((entry[0], entry[1]))
        results.append(entry)

    os.makedirs(exports_path, exist_ok=True)

    for metrics in results:
        name = metrics[0]
        plt.plot(metrics[1], metrics[2], label=name)

    plt.legend(loc="best")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(
        f"{exports_path}/PSO(dimension = {dimension}).svg",
        format="svg",
    )
