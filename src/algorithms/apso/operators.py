import typing as t
from algorithms.apso.individual import Individual

T = t.TypeVar("T")
U = t.TypeVar("U")


class BaseTwoOptOperator(t.Generic[T, U]):
    def run(self, individual: Individual[T, U]) -> Individual[T, U]:
        return individual


class BasePathRelinkingOperator(t.Generic[T, U]):
    def run(self, individual: Individual[T, U], best_individual: Individual[T, U]) -> Individual[T, U]:
        return individual


class BaseSwapOperator(t.Generic[T, U]):
    def run(self, individual: Individual[T, U]) -> Individual[T, U]:
        return individual
