import typing as t
from dataclasses import dataclass

T = t.TypeVar("T")
U = t.TypeVar("U")
V = t.TypeVar("V")

@dataclass
class CombinatorialFunctionDefinition:
    name: str
    description: str
    function: t.Callable[[t.List[T]], U]
    target: t.Union[t.Literal["maximise"], t.Literal["minimise"]]
    values: t.List[T]
    costs: t.Optional[V]
