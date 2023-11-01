import os
import typing as t
import numpy as np
import matplotlib.pyplot as plt


def graph_hillclimber(
    name: str, y_label: str, metrics: t.List[t.Tuple[np.uint64, np.float32]], show=False
):
    x, y = zip(*metrics)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)

    plt.title(name)
    plt.xlabel("Generations")
    plt.ylabel(f"Best {y_label}")

    os.makedirs("./outputs", exist_ok=True)
    plt.savefig(f"./outputs/{name}.svg", format="svg")

    if show:
        plt.show()


def graph_genetic_algorithm(
    name: str, metrics: t.List[t.Tuple[np.uint64, t.List[np.float32]]], show=False
):
    x, ys = zip(*metrics)

    plt.figure(figsize=(10, 6))
    plt.plot(x, ys, "o")

    plt.title(name)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    os.makedirs("./outputs", exist_ok=True)
    plt.savefig(f"./outputs/{name}.svg", format="svg")

    if show:
        plt.show()
