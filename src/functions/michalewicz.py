import typing as t
import numpy as np


def michalewicz(x: t.List[np.float32]):
    m = 10
    return -np.sum(
        [np.sin(x[i]) * (np.sin(i * x[i] ** 2 / np.pi)) for i in range(len(x))]
        ** (2 * m)
    )


def run_michalewicz(dimensions: int):
    pass
