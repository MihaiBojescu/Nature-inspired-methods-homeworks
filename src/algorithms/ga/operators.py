import typing as t
from algorithms.ga.individual import Individual

T = t.TypeVar("T")


class BaseSelectionOperator(t.Generic[T]):
    def run(self, population: t.List[Individual[T]]) -> t.List[Individual[T]]:
        return population


class BaseCrossoverOperator(t.Generic[T]):
    def run(self, parent_1: Individual[T], parent_2: Individual[T]) -> t.Tuple[Individual[T], Individual[T]]:
        return parent_1, parent_2


class BaseMutationOperator(t.Generic[T]):
    def run(self, child: Individual[T]) -> Individual[T]:
        return child
