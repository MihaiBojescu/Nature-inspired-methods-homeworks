import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.pso.individual import Individual
from util.sort import quicksort


class ParticleSwarmOptimisation:
    _population: t.List[Individual]
    _generation: np.uint64
    _fitness_compare_function: t.Callable[[np.float32, np.float32], bool]
    _criteria_function: t.Callable[[t.List[np.float32], np.float32, np.uint64], bool]

    __debug: bool

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
        debug: bool = False,
    ) -> None:
        self._population = [
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
        self._generation = np.uint64(0)
        self._fitness_compare_function = fitness_compare_function
        self._criteria_function = criteria_function

        self.__debug = debug

        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )

    def run(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        best_individual = self._population[0]

        while not self._criteria_function(
            best_individual.position, best_individual.fitness, self._generation
        ):
            self.step()

        best_individual = self._population[0]

        return best_individual.position, best_individual.fitness, self._generation

    def step(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        self._print(self._generation)
        best_individual = self._population[0]

        for individual in self._population:
            individual.update(team_best_position=best_individual.personal_best_position)

        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )
        best_individual = self._population[0]

        self._generation += 1

        return best_individual.position, best_individual.fitness, self._generation

    def _print(self, generation: int) -> None:
        if not self.__debug:
            return

        best_individual = self._population[0]
        print(
            f"Particle swarm optimisation algorithm generation {generation}: {best_individual.fitness}"
        )
