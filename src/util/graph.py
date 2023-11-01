import typing as t
import numpy as np
import matplotlib.pyplot as plt

def graph_hillclimber(name: str, metrics: t.List[t.Tuple[np.uint64, np.float32]]):
    x, y = zip(*metrics)

    plt.figure(figsize=(10,6))
    plt.plot(x, y)

    plt.title(name)
    plt.xlabel('Generations')
    plt.ylabel('Best X')

    plt.savefig(f"{name}.svg", format="svg")
    plt.show()

