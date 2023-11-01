import typing as t
import numpy as np


def h1p(x: np.uint32):
    return x**3 - 60 * (x**2) + 900 * x + 100


def rastrigin(n: t.List[np.float32]):
    return lambda x: 10 * n + np.sum(
        [x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(len(x))]
    )


def griewangk(x: t.List[np.float32]):
    return (
        np.sum([x[i] ** 2 / 4000 for i in range(len(x))])
        - np.prod([np.cos(x[i] / np.sqrt[i]) for i in range(len(x))])
        + 1
    )


def rosenbrock_valley(x: t.List[np.float32]):
    return np.sum(
        [100 * (x[i] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x))]
    )


def michalewicz(x: t.List[np.float32]):
    m = 10
    return -np.sum(
        [np.sin(x[i]) * (np.sin(i * x[i] ** 2 / np.pi)) for i in range(len(x))]
        ** (2 * m)
    )
