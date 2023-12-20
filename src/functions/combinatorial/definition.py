import typing as t
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

T = t.TypeVar("T")
U = t.TypeVar("U")

@dataclass
class CombinatorialFunctionDefinition:
    name: str
    description: str
    function: t.Callable[[t.List[T]], U]
    target: t.Union[t.Literal["maximise"], t.Literal["minimise"]]
    values: t.List[T]
