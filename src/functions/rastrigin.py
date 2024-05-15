import numpy as np
import numpy.typing as npt
from functions.definition import FunctionDefinition, FunctionBoundaries


def rastrigin(x: npt.NDArray[np.float32]):
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


rastrigin_definition = FunctionDefinition(
    name="Rastrigin",
    function=rastrigin,
    target="minimise",
    value_boundaries=FunctionBoundaries(min=-5.12, max=5.12),
    best_result=0,
)
