import typing as t
import numpy as np
from functions.definition import FunctionDefinition, FunctionBoundaries


def rastrigin(x: t.List[np.float32]):
    A = 10
    n = len(x)
    return A * n + np.sum(
        [x[i] ** 2 - A * np.cos(2 * np.pi * x[i]) for i in range(len(x))]
    )


rastrigin_definition = FunctionDefinition(
    name="Rastrigin",
    function=rastrigin,
    target="minimise",
    value_boundaries=FunctionBoundaries(min=-5.12, max=5.12),
    best_result=0,
)
