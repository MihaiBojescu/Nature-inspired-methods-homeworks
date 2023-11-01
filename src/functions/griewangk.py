import typing as t
import numpy as np


def griewangk(x: t.List[np.float32]):
    return (
        np.sum([x[i] ** 2 / 4000 for i in range(len(x))])
        - np.prod([np.cos(x[i] / np.sqrt[i]) for i in range(len(x))])
        + 1
    )


def run_griewangk(dimensions: int):
    pass
