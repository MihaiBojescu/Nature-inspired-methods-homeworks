import typing as t
import numpy as np
from functions.definition import FunctionDefinition, FunctionBoundaries


def michalewicz(x: t.List[np.float32]):
    m = 10
    return -np.sum(
        [
            np.sin(x[i]) * (np.sin(i * x[i] ** 2 / np.pi)) ** (2 * m)
            for i in range(len(x))
        ]
    )


michalewicz_definition = FunctionDefinition(
    name="Michalewicz",
    function=michalewicz,
    target="minimise",
    value_boundaries=FunctionBoundaries(min=0, max=np.pi),
    best_result=0,
)
