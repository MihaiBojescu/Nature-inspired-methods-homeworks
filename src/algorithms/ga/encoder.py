import typing as t
import numpy as np

T = t.TypeVar("T")


class BaseEncoder:
    def encode(self, value: t.List[np.int64]) -> T:
        pass

    def decode(self, value: T) -> t.List[np.int64]:
        pass
