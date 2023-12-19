import typing as t
import numpy as np
import numpy.typing as npt


class Individual:
    __x: npt.NDArray[np.float32]
    __personal_best_x: npt.NDArray[np.float32]
    __fitness: np.float32
    __personal_best_fitness: np.float32
    __velocity: npt.NDArray[np.float32]
    __inertia_weight: np.float32
    __cognitive_parameter: np.float32
    __social_parameter: np.float32
    __intensification_parameter: np.float32
    __fitness_function: t.Callable[[npt.NDArray[np.float32]], np.float32]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]

    def __init__(
        self,
        initial_position: npt.NDArray[np.float32],
        inertia_weight: np.float32,
        cognitive_parameter: np.float32,
        social_parameter: np.float32,
        intensification_parameter: np.float32,
        fitness_function: t.Callable[[npt.NDArray[np.float32]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
    ) -> None:
        self.__x = np.array(initial_position)
        self.__velocity = np.array(
            [np.random.uniform(low=-1, high=1) for _ in range(len(self.__x))]
        )
        self.__personal_best_x = np.copy(initial_position)
        self.__inertia_weight = inertia_weight
        self.__cognitive_parameter = cognitive_parameter
        self.__social_parameter = social_parameter
        self.__intensification_parameter = intensification_parameter
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

        self.__fitness = self.__fitness_function(self.__x)
        self.__personal_best_fitness = self.__fitness

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

    def update(self, team_best_position: npt.NDArray[np.float32]) -> None:
        y = self.__calculate_initial_y(team_best_position)

        self.__velocity = self.__velocity * self.__inertia_weight
        self.__velocity += (
            (-1 - y) * self.__cognitive_parameter * np.random.uniform(low=-1, high=1)
        )
        self.__velocity += (
            (1 - y) * self.__social_parameter * np.random.uniform(low=-1, high=1)
        )

        lambda_t = self.__velocity + y
        y = self.__calculate_y_by_lambda(lambda_t)

        self.__x = self.__calculate_x_by_y(
            y,
            team_best_position[i],
        )
        self.__fitness = self.__fitness_function(self.__x)

        if self.__fitness_compare_function(
            self.__fitness, self.__personal_best_fitness
        ):
            self.__personal_best_x = self.__x
            self.__personal_best_fitness = self.__fitness

    def __calculate_initial_y(
        self, team_best_x: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.int8]:
        y = np.zeros(self.__x.shape, dtype=np.int8)

        for i, _ in enumerate(y):
            if (
                self.__x[i] == self.__personal_best_x[i]
                and self.__x[i] == team_best_x[i]
            ):
                y[i] = np.random.choice([-1, 1], size=1)
                continue

            if self.__x[i] == self.__personal_best_x[i]:
                y[i] = -1
                continue

            if self.__x[i] == team_best_x[i]:
                y[i] = 1
                continue

        return y

    def __calculate_y_by_lambda(
        self, lambda_t: npt.NDArray[np.int8]
    ) -> npt.NDArray[np.int8]:
        y = np.zeros(self.__x.shape, dtype=np.int8)

        for i, _ in enumerate(y):
            if lambda_t[i] > self.__intensification_parameter:
                y[i] = 1
                continue

            if lambda_t[i] < -self.__intensification_parameter:
                y[i] = -1
                continue

        return y

    def __calculate_x_by_y(
        self,
        y: npt.NDArray[np.int8],
        team_best_x: npt.NDArray[np.uint32],
    ) -> int:
        x = np.zeros(self.__x.shape)

        for i, _ in enumerate(x):
            if y[i] == 1:
                x[i] = team_best_x
                continue

            if y[i] == -1:
                x[i] = self.__personal_best_x
                continue

            x[i] = np.random.randint(low=0, high=len(x))

        return x
