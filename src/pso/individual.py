import typing as t
import numpy as np
import numpy.typing as npt


class Individual:
    __position: npt.NDArray[np.float32]
    __personal_best_position: npt.NDArray[np.float32]
    __velocity: npt.NDArray[np.float32]
    __inertia_bias: np.float32
    __personal_best_position_bias: np.float32
    __team_best_position_bias: np.float32
    __random_jitter_bias: np.float32
    __personal_best_comparator: t.Callable[
        [npt.NDArray[np.float32], npt.NDArray[np.float32]], npt.NDArray[np.float32]
    ]

    def __init__(
        self,
        initial_position: npt.NDArray[np.float32],
        inertia_bias: np.float32,
        personal_best_position_bias: np.float32,
        team_best_position_bias: np.float32,
        random_jitter_bias: np.float32,
        personal_best_comparator: t.Callable[
            [npt.NDArray[np.float32], npt.NDArray[np.float32]], npt.NDArray[np.float32]
        ] = lambda a, b: a
        < b,
    ) -> None:
        self.__position = initial_position
        self.__velocity = np.array(
            [np.random.uniform(low=-1, high=1) for _ in range(self.__position)]
        )
        self.__personal_best_position = initial_position
        self.__inertia_bias = inertia_bias
        self.__personal_best_position_bias = personal_best_position_bias
        self.__team_best_position_bias = team_best_position_bias
        self.__random_jitter_bias = random_jitter_bias
        self.__personal_best_comparator = personal_best_comparator

    @property
    def position(self) -> npt.NDArray[np.float32]:
        return self.__position

    @property
    def velocity(self) -> npt.NDArray[np.float32]:
        return self.__velocity

    @property
    def personal_best_position(self) -> npt.NDArray[np.float32]:
        return self.__personal_best_position

    def update(self, team_best_position: npt.NDArray[np.float32]) -> None:
        self.__velocity = self.__velocity * self.__inertia_bias
        self.__velocity += (
            self.__personal_best_position * self.__personal_best_position_bias
        )
        self.__velocity += team_best_position * self.__team_best_position_bias
        self.__velocity += np.array(
            [
                self.__random_jitter_bias * np.random.uniform(low=-1, high=1)
                for _ in range(len(self.__position))
            ]
        )

        self.__position += self.__velocity

        if self.__personal_best_comparator(
            self.__position, self.__personal_best_position
        ):
            self.__personal_best_position = self.__position
