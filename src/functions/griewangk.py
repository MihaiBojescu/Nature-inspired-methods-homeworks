import typing as t
import numpy as np
from functions.definition import FunctionDefinition, FunctionBoundaries


def griewangk(x: t.List[np.float32]):
    sum_value = np.sum([(x[i] ** 2) / 4000 for i in range(len(x))])
    prod_value = np.prod([np.cos(x[i] / np.sqrt(i + 1)) for i in range(len(x))])

    return sum_value - prod_value + 1


griewangk_definition = FunctionDefinition(
    name="Griewangk",
    function=griewangk,
    target="minimise",
    value_boundaries=FunctionBoundaries(min=-600.0, max=600.0),
    best_result=0,
)
