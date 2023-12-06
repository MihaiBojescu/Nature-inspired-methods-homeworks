import typing as t
from dataclasses import dataclass

T = t.TypeVar("T")

@dataclass
class FunctionBoundaries:
    min: T
    max: T

@dataclass
class FunctionDefinition:
    name: str
    function: t.Callable[[T], float]
    target: t.Union[t.Literal["maximise"], t.Literal["minimise"]]
    value_boundaries: FunctionBoundaries
    best_result: T
