import typing as t
import numpy as np


class BaseSelectionOperator:
    def run(self, population: t.List[t.List[np.int64]]) -> t.List[t.List[np.int64]]:
        return population


class BaseCrossoverOperator:
    def run(
        self, parent_1: t.List[np.int64], parent_2: t.List[np.int64]
    ) -> t.Tuple[t.List[np.int64], t.List[np.int64]]:
        return parent_1, parent_2


class BaseMutationOperator:
    def run(self, child: t.List[np.int64]) -> t.List[np.int64]:
        return child


class SwapMutationOperator(BaseMutationOperator):
    def run(self, child: t.List[np.int64]) -> t.List[np.int64]:
        [a, b] = np.random.choice(len(child), size=2, replace=False)
        child[a], child[b] = child[b], child[a]

        return child
