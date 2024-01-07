import random
import typing as t
from algorithms.apso.operators import (
    BasePathLinkerOperator,
    BaseSwapOperator,
    BaseTwoOptOperator,
)

T = t.TypeVar("T")


class Individual(t.Generic[T]):
    __position: T
    __personal_best_position: T
    __fitness: float
    __personal_best_fitness: float
    __fitness_function: t.Callable[[T], float]
    __fitness_compare_function: t.Callable[[float, float], bool]

    __two_opt_operator: BaseTwoOptOperator[T]
    __path_linker_operator: BasePathLinkerOperator[T]
    __swap_operator: BaseSwapOperator[T]
    __two_opt_operator_probability: float
    __path_linker_operator_probability: float
    __swap_operator_probability: float

    def __init__(
        self,
        initial_position: T,
        fitness_function: t.Callable[[T], float],
        fitness_compare_function: t.Callable[[float, float], bool],
        two_opt_operator: BaseTwoOptOperator[T],
        path_linker_operator: BasePathLinkerOperator[T],
        swap_operator: BaseSwapOperator[T],
        two_opt_operator_probability: float,
        path_linker_operator_probability: float,
        swap_operator_probability: float,
    ) -> None:
        self.__position = initial_position
        self.__personal_best_position = initial_position.copy()
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

        self.__fitness = self.__fitness_function(self.__position)
        self.__personal_best_fitness = self.__fitness

        self.__two_opt_operator = two_opt_operator
        self.__path_linker_operator = path_linker_operator
        self.__swap_operator = swap_operator
        self.__two_opt_operator_probability = two_opt_operator_probability
        self.__path_linker_operator_probability = path_linker_operator_probability
        self.__swap_operator_probability = swap_operator_probability

    @property
    def position(self) -> T:
        return self.__position

    @property
    def personal_best_position(self) -> T:
        return self.__personal_best_position

    @property
    def fitness(self) -> float:
        return self.__fitness

    @property
    def personal_best_fitness(self) -> float:
        return self.__personal_best_fitness

    def update(self) -> None:
        match random.choices(
            [1, 2, 3],
            weights=[
                self.__two_opt_operator_probability,
                self.__path_linker_operator_probability,
                self.__swap_operator_probability,
            ],
            k=1,
        )[0]:
            case 1:
                self.__position = self.__two_opt_operator.run(self.__position)
            case 2:
                self.__position = self.__path_linker_operator.run(self.__position)
            case 3:
                self.__position = self.__swap_operator.run(self.__position)

        self.__fitness = self.__fitness_function(self.__position)

        if self.__fitness_compare_function(self.__fitness, self.__personal_best_fitness):
            self.__personal_best_position = self.__position
            self.__personal_best_fitness = self.__fitness
