import typing as t
from algorithms.apso.individual import Individual

T = t.TypeVar("T")
U = t.TypeVar("U")


class BaseTwoOptOperator(t.Generic[T, U]):
    def run(self, values: Individual[T, U]) -> T:
        return values


class BasePathLinkerOperator(t.Generic[T, U]):
    def run(self, values: Individual[T, U]) -> T:
        return values


class BaseSwapOperator(t.Generic[T, U]):
    def run(self, values: Individual[T, U]) -> T:
        return values
