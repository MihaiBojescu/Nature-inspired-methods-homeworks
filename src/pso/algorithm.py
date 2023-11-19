import typing as t
import numpy as np
import numpy.typing as npt
from pso.individual import Individual
from util.sort import quicksort


class Algorithm:
    __population: t.List[Individual]
    __generation: np.uint64
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]
    __criteria_function: t.Callable[[t.List[np.float32], np.float32, np.uint64], bool]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], npt.NDArray[np.float32]],
        fitness_function: t.Callable[[np.float32], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        criteria_function: t.Callable[
            [t.List[np.float32], np.float32, np.uint64], bool
        ],
        inertia_bias: np.float32,
        personal_best_position_bias: np.float32,
        team_best_position_bias: np.float32,
        random_jitter_bias: np.float32,
    ) -> None:
        self.__population = [
            Individual(
                initial_position=position,
                inertia_bias=inertia_bias,
                personal_best_position_bias=personal_best_position_bias,
                team_best_position_bias=team_best_position_bias,
                random_jitter_bias=random_jitter_bias,
                fitness_function=fitness_function,
                fitness_compare_function=fitness_compare_function,
            )
            for position in generate_initial_population()
        ]
        self.__generation = np.uint64(0)
        self.__fitness_compare_function = fitness_compare_function
        self.__criteria_function = criteria_function

        self.__population = quicksort(
            data=self.__population,
            comparator=lambda a, b: self.__fitness_compare_function(
                a.fitness, b.fitness
            ),
        )

    def run(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        best_individual = self.__population[0]

        while self.__criteria_function(
            best_individual.position, best_individual.fitness, self.__generation
        ):
            self.step()

        best_individual = self.__population[0]

        return best_individual.position, best_individual.fitness, self.__generation

    def step(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        best_individual = self.__population[0]

        for individual in self.__population:
            individual.update(team_best_position=best_individual.personal_best_position)

        self.__population = quicksort(
            data=self.__population,
            comparator=lambda a, b: self.__fitness_compare_function(
                a.fitness, b.fitness
            ),
        )
        best_individual = self.__population[0]

        self.__generation += 1

        return best_individual.position, best_individual.fitness, self.__generation
