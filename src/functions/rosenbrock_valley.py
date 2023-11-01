import typing as t
import numpy as np


def rosenbrock_valley(x: t.List[np.float32]):
    return np.sum(
        [100 * (x[i] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x))]
    )


def run_rosenbrock_valley(dimensions: int):
    pass
