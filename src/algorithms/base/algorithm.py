import typing as t
import numpy as np

T = t.TypeVar("T")

class BaseAlgorithm:
    def run(self) -> t.Tuple[T, np.float32, np.uint64]:
        pass

    def step(self) -> t.Tuple[T, np.float32, np.uint64]:
        pass
