import typing as t

T = t.TypeVar("T")
U = t.TypeVar("U")

class Individual(t.Generic[T, U]):
    __genes: T
    __fitness: U
    __fitness_function: t.Callable[[T], U]

    def __init__(
        self,
        genes: T,
        fitness_function: t.Callable[[T], U],
    ) -> None:
        self.__genes = genes
        self.__fitness_function = fitness_function
        self.__fitness = self.__fitness_function(self.__genes)

    @staticmethod
    def from_genes(genes: T, fitness_function: t.Callable[[T], U]) -> t.Self:
        return Individual(genes=genes, fitness_function=fitness_function)

    @property
    def genes(self) -> T:
        return self.__genes

    @genes.setter
    def genes(self, value: T):
        self.__genes = value
        self.__fitness = self.__fitness_function(self.__genes)

    @property
    def fitness(self) -> U:
        return self.__fitness
