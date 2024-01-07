import typing as t
import numpy as np

T = t.TypeVar("T")


class BaseAlgorithm(t.Generic[T]):
    @property
    def name(self) -> str:
        return "Base algorithm"

    def run(self) -> t.Tuple[T, float, int]:
        pass

    def step(self) -> t.Tuple[T, float, int]:
        pass
