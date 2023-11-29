import typing as t
import numpy as np
import numpy.typing as npt


class Individual:
    __position: npt.NDArray[np.float32]
    __personal_best_position: npt.NDArray[np.float32]
    __fitness: np.float32
    __personal_best_fitness: np.float32
    __velocity: npt.NDArray[np.float32]
    __inertia_weight: np.float32
    __cognitive_parameter: np.float32
    __cognitive_random_parameter: np.float32
    __social_parameter: np.float32
    __social_random_parameter: np.float32
    __random_jitter_parameter: np.float32
    __random_jitter_random_parameter: np.float32
    __fitness_function: t.Callable[[npt.NDArray[np.float32]], np.float32]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]

    def __init__(
        self,
        initial_position: npt.NDArray[np.float32],
        inertia_weight: np.float32,
        cognitive_parameter: np.float32,
        social_parameter: np.float32,
        random_jitter_parameter: np.float32,
        fitness_function: t.Callable[[npt.NDArray[np.float32]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
    ) -> None:
        self.__position = np.array(initial_position)
        self.__velocity = np.array(
            [np.random.uniform(low=-1, high=1) for _ in range(len(self.__position))]
        )
        self.__personal_best_position = np.array(initial_position)
        self.__inertia_weight = inertia_weight
        self.__cognitive_parameter = cognitive_parameter
        self.__cognitive_random_parameter = np.random.uniform(low=0.0, high=1.0)
        self.__social_parameter = social_parameter
        self.__social_random_parameter = np.random.uniform(low=0.0, high=1.0)
        self.__random_jitter_parameter = random_jitter_parameter
        self.__random_jitter_random_parameter = np.random.uniform(low=0.0, high=1.0)
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

        self.__fitness = self.__fitness_function(self.__position)
        self.__personal_best_fitness = self.__fitness

    @property
    def position(self) -> npt.NDArray[np.float32]:
        return self.__position

    @property
    def personal_best_position(self) -> npt.NDArray[np.float32]:
        return self.__personal_best_position

    @property
    def fitness(self) -> np.float32:
        return self.__fitness

    @property
    def personal_best_fitness(self) -> np.float32:
        return self.__personal_best_fitness

    def update(self, team_best_position: npt.NDArray[np.float32]) -> None:
        self.__velocity = self.__velocity * self.__inertia_weight
        self.__velocity += (
            (self.__personal_best_position - self.__position)
            * self.__cognitive_parameter
            * self.__cognitive_random_parameter
        )
        self.__velocity += (
            (team_best_position - self.__position)
            * self.__social_parameter
            * self.__social_random_parameter
        )
        self.__velocity += np.array(
            [
                np.random.uniform(low=-1, high=1)
                * self.__random_jitter_parameter
                * self.__random_jitter_random_parameter
                for _ in range(len(self.__position))
            ]
        )

        self.__position += self.__velocity
        self.__fitness = self.__fitness_function(self.__position)

        if self.__fitness_compare_function(
            self.__fitness, self.__personal_best_fitness
        ):
            self.__personal_best_position = self.__position
            self.__personal_best_fitness = self.__fitness
