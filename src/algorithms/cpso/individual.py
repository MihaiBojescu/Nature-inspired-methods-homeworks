import copy
import random
import typing as t
import numpy as np

T = t.TypeVar("T")


class Individual:
    __position: t.List[int]
    __personal_best_position: t.List[int]
    __fitness: float
    __personal_best_fitness: float
    __velocity: t.List[float]
    __inertia_weight: float
    __cognitive_parameter: float
    __social_parameter: float
    __intensification_parameter: float
    __fitness_function: t.Callable[[t.List[float]], float]
    __fitness_compare_function: t.Callable[[float, float], bool]

    def __init__(
        self,
        initial_position: t.List[int],
        inertia_weight: float,
        cognitive_parameter: float,
        social_parameter: float,
        intensification_parameter: float,
        fitness_function: t.Callable[[t.List[float]], float],
        fitness_compare_function: t.Callable[[float, float], bool],
    ) -> None:
        self.__position = initial_position
        self.__velocity = [random.uniform(a=-1, b=1) for _ in range(len(self.__position))]
        self.__personal_best_position = copy.copy(initial_position)
        self.__inertia_weight = inertia_weight
        self.__cognitive_parameter = cognitive_parameter
        self.__social_parameter = social_parameter
        self.__intensification_parameter = intensification_parameter
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

        self.__fitness = self.__fitness_function(self.__position)
        self.__personal_best_fitness = self.__fitness

    @property
    def position(self) -> t.List[int]:
        return self.__position

    @property
    def personal_best_position(self) -> t.List[float]:
        return self.__personal_best_position

    @property
    def fitness(self) -> float:
        return self.__fitness

    @property
    def personal_best_fitness(self) -> float:
        return self.__personal_best_fitness

    def update(self, team_best_position: t.List[float]) -> None:
        y = self.__calculate_initial_y(team_best_position)

        self.__velocity = self.__velocity * self.__inertia_weight
        self.__velocity += (-1 - y) * self.__cognitive_parameter * np.random.uniform(low=-1, high=1)
        self.__velocity += (1 - y) * self.__social_parameter * np.random.uniform(low=-1, high=1)

        lambda_t = self.__velocity + y
        y = self.__calculate_y_by_lambda(lambda_t)

        self.__position = self.__calculate_x_by_y(
            y,
            team_best_position,
        )
        self.__fitness = self.__fitness_function(self.__position)

        if self.__fitness_compare_function(self.__fitness, self.__personal_best_fitness):
            self.__personal_best_position = self.__position
            self.__personal_best_fitness = self.__fitness

    def __calculate_initial_y(self, team_best_x: t.List[np.uint32]) -> t.List[int]:
        y = np.zeros(self.__position.shape, dtype=int)

        for i, _ in enumerate(y):
            if self.__position[i] == self.__personal_best_position[i] and self.__position[i] == team_best_x[i]:
                y[i] = random.choice([-1, 1])
                continue

            if self.__position[i] == self.__personal_best_position[i]:
                y[i] = -1
                continue

            if self.__position[i] == team_best_x[i]:
                y[i] = 1
                continue

        return y

    def __calculate_y_by_lambda(self, lambda_t: t.List[int]) -> t.List[int]:
        y = [0 for _ in range(len(self.__position))]

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
        y: t.List[int],
        team_best_position: t.List[int],
    ) -> int:
        x = np.zeros(self.__position.shape, dtype=self.__position.dtype)

        for i, _ in enumerate(x):
            if y[i] == 1:
                x[i] = team_best_position[i]
                continue

            if y[i] == -1:
                x[i] = self.__personal_best_position[i]
                continue

            x[i] = np.random.randint(low=0, high=len(x))

        return x
