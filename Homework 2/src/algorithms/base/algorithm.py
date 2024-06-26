import typing as t
import numpy as np

T = t.TypeVar("T")

class BaseAlgorithm:
    @property
    def name(self) -> str:
        return "Base algorithm"

    def run(self) -> t.Tuple[T, np.float32, np.uint64]:
        pass

    def step(self) -> t.Tuple[T, np.float32, np.uint64]:
        pass
