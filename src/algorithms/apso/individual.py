import typing as t

T = t.TypeVar("T")
U = t.TypeVar("U")


class Individual(t.Generic[T, U]):
    __position: T
    __fitness: U
    __fitness_function: t.Callable[[T], U]

    def __init__(
        self,
        initial_position: T,
        fitness_function: t.Callable[[T], U],
    ) -> None:
        self.__position = initial_position
        self.__fitness_function = fitness_function
        self.__fitness = self.__fitness_function(self.__position)

    @property
    def position(self) -> T:
        return self.__position

    @position.setter
    def position(self, position: T):
        self.__position = position
        self.__fitness = self.__fitness_function(self.__position)

    @property
    def fitness(self) -> U:
        return self.__fitness
