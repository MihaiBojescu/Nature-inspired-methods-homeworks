import typing as t

T = t.TypeVar("T")
U = t.TypeVar("U")


class BaseAlgorithm(t.Generic[T, U]):
    @property
    def name(self) -> str:
        return "Base algorithm"

    def run(self) -> t.Tuple[T, U, int]:
        pass

    def step(self) -> t.Tuple[T, U, int]:
        pass
