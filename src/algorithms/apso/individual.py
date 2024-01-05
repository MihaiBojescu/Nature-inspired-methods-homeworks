import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.apso.operators import TwoOptOperator, PathLinkerOperator, SwapOperator


class Individual:
    __x: t.List[t.List[np.int64]]
    __personal_best_x: t.List[t.List[np.int64]]
    __fitness: np.float32
    __personal_best_fitness: np.float32
    __fitness_function: t.Callable[[t.List[t.List[np.int64]]], np.float32]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]

    __two_opt_operator_probability: float
    __path_linker_operator_probability: float
    __swap_operator_probability: float
    __two_opt_operator: TwoOptOperator
    __path_linker_operator: PathLinkerOperator
    __swap_operator: SwapOperator

    def __init__(
        self,
        initial_position: t.List[t.List[np.int64]],
        fitness_function: t.Callable[[t.List[t.List[np.int64]]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        two_opt_operator_probability: float,
        path_linker_operator_probability: float,
        swap_operator_probability: float,
    ) -> None:
        self.__x = initial_position
        self.__personal_best_x = initial_position.copy()
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

        self.__fitness = self.__fitness_function(self.__x)
        self.__personal_best_fitness = self.__fitness

        self.__two_opt_operator_probability = two_opt_operator_probability
        self.__path_linker_operator_probability = path_linker_operator_probability
        self.__swap_operator_probability = swap_operator_probability
        self.__two_opt_operator = TwoOptOperator(
            fitness_function=self.__fitness_function, fitness_compare_function=self.__fitness_compare_function
        )
        self.__path_linker_operator = PathLinkerOperator(
            fitness_function=self.__fitness_function, fitness_compare_function=self.__fitness_compare_function
        )
        self.__swap_operator = SwapOperator()

    @property
    def position(self) -> t.List[t.List[np.int64]]:
        return self.__x

    @property
    def personal_best_position(self) -> t.List[t.List[np.int64]]:
        return self.__personal_best_x

    @property
    def fitness(self) -> np.float32:
        return self.__fitness

    @property
    def personal_best_fitness(self) -> np.float32:
        return self.__personal_best_fitness

    def update(self) -> None:
        match np.random.choice(
            [1, 2, 3],
            p=[
                self.__two_opt_operator_probability,
                self.__path_linker_operator_probability,
                self.__swap_operator_probability,
            ],
            size=1,
            replace=False,
        ):
            case 1:
                self.__x = self.__two_opt_operator.run(self.__x)
            case 2:
                self.__x = self.__path_linker_operator.run(self.__x)
            case 3:
                self.__x = self.__swap_operator.run(self.__x)

        self.__fitness = self.__fitness_function(self.__x)

        if self.__fitness_compare_function(self.__fitness, self.__personal_best_fitness):
            self.__personal_best_x = self.__x
            self.__personal_best_fitness = self.__fitness
