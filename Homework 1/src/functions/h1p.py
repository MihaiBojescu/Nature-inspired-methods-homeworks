import numpy as np
import numpy.typing as npt
from functions.definition import FunctionDefinition, FunctionBoundaries


def h1p(x: npt.NDArray[np.float32]):
    return x**3 - 60 * (x**2) + 900 * x + 100


h1p_definition = FunctionDefinition(
    name="H1'",
    function=h1p,
    target="maximise",
    value_boundaries=FunctionBoundaries(min=0.0, max=31.0),
    best_result=4100,
)
