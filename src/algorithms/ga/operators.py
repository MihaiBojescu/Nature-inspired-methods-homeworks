import typing as t
from algorithms.ga.individual import Individual

T = t.TypeVar("T")
U = t.TypeVar("U")


class BaseSelectionOperator(t.Generic[T, U]):
    def run(self, population: t.List[Individual[T, U]]) -> t.List[Individual[T, U]]:
        return population


class BaseCrossoverOperator(t.Generic[T, U]):
    def run(
        self, parent_1: Individual[T, U], parent_2: Individual[T, U]
    ) -> t.Tuple[Individual[T, U], Individual[T, U]]:
        return parent_1, parent_2


class BaseMutationOperator(t.Generic[T, U]):
    def run(self, child: Individual[T, U]) -> Individual[T, U]:
        return child
