import typing as t
import numpy as np

T = t.TypeVar("T")
DecodedIndividual = t.Tuple[T, np.float32]


class Individual(t.Generic[T]):
    __genes: T
    __fitness: np.float32
    __fitness_function: t.Callable[[T], np.float32]

    def __init__(
        self,
        genes: T,
        fitness_function: t.Callable[[T], np.float32],
    ) -> None:
        self.__genes = genes
        self.__fitness = fitness_function(genes)
        self.__fitness_function = fitness_function

    @staticmethod
    def from_genes(genes: T, fitness_function: t.Callable[[T], np.float32]) -> t.Self[T]:
        return Individual(genes=genes, fitness_function=fitness_function)

    @property
    def genes(self) -> T:
        return self.__genes

    @genes.setter
    def genes(self, value: T):
        self.__genes = value
        self.__fitness = self.__fitness_function(self.__genes)

    @property
    def fitness(self) -> np.float32:
        return self.__fitness
