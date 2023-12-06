import typing as t
import numpy as np
from functions.definition import FunctionDefinition, FunctionBoundaries


def rosenbrock(x: t.List[np.float32]):
    return np.sum(
        [100 * (x[i] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x))]
    )


rosenbrock_definition = FunctionDefinition(
    name="Rosenbrock",
    function=rosenbrock,
    target="minimise",
    value_boundaries=FunctionBoundaries(min=-2.048, max=2.048),
    best_result=0,
)
