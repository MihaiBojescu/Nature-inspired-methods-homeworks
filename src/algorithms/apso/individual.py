import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.apso.operators import TwoOptOperator, PathLinkerOperator, SwapOperator


class Individual:
    __x: npt.NDArray[np.float32]
    __personal_best_x: npt.NDArray[np.float32]
    __fitness: np.float32
    __personal_best_fitness: np.float32
    __fitness_function: t.Callable[[npt.NDArray[np.float32]], np.float32]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]

    __two_opt_operator: TwoOptOperator
    __path_linker_operator: PathLinkerOperator
    __swap_operator: SwapOperator

    def __init__(
        self,
        initial_position: npt.NDArray[np.float32],
        fitness_function: t.Callable[[npt.NDArray[np.float32]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
    ) -> None:
        self.__x = np.array(initial_position)
        self.__personal_best_x = np.copy(initial_position)
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

        self.__fitness = self.__fitness_function(self.__x)
        self.__personal_best_fitness = self.__fitness

        self.__two_opt_operator = TwoOptOperator(
            fitness_function=self.__fitness_function, fitness_compare_function=self.__fitness_compare_function
        )
        self.__path_linker_operator = PathLinkerOperator(
            fitness_function=self.__fitness_function, fitness_compare_function=self.__fitness_compare_function
        )
        self.__swap_operator = SwapOperator()

    @property
    def position(self) -> npt.NDArray[np.float32]:
        return self.__x

    @property
    def personal_best_position(self) -> npt.NDArray[np.float32]:
        return self.__personal_best_x

    @property
    def fitness(self) -> np.float32:
        return self.__fitness

    @property
    def personal_best_fitness(self) -> np.float32:
        return self.__personal_best_fitness

    def update(self) -> None:
        match np.random.randint(low=1, high=4):
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
