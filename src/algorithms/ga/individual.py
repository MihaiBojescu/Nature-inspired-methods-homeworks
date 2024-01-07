import typing as t

T = t.TypeVar("T")
DecodedIndividual = t.Tuple[T, float]


class Individual(t.Generic[T]):
    __genes: T
    __fitness: float
    __fitness_function: t.Callable[[T], float]

    def __init__(
        self,
        genes: T,
        fitness_function: t.Callable[[T], float],
    ) -> None:
        self.__genes = genes
        self.__fitness = fitness_function(genes)
        self.__fitness_function = fitness_function

    @staticmethod
    def from_genes(genes: T, fitness_function: t.Callable[[T], float]) -> t.Self:
        return Individual(genes=genes, fitness_function=fitness_function)

    @property
    def genes(self) -> T:
        return self.__genes

    @genes.setter
    def genes(self, value: T):
        self.__genes = value
        self.__fitness = self.__fitness_function(self.__genes)

    @property
    def fitness(self) -> float:
        return self.__fitness
