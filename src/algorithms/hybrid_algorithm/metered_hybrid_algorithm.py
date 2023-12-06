import time
import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.hybrid_algorithm.hybrid_algorithm import HybridAlgorithm
from algorithms.binary_genetic_algorithm.individual import DecodedIndividual
from algorithms.binary_genetic_algorithm.selection_functions import (
    roulette_wheel_selection,
)
from functions.definition import FunctionDefinition
from util.sort import maximise, minimise, quicksort

T = t.TypeVar("T")


class MeteredHybridAlgorithm(HybridAlgorithm):
    _metrics_runtime: t.List[t.Tuple[np.uint64, np.uint64]]
    _metrics_values: t.List[t.Tuple[np.uint64, any]]
    _metrics_fitness: t.List[t.Tuple[np.uint64, np.float32]]

    def __init__(
        self,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        generate_initial_population: t.Callable[[], t.List[any]],
        fitness_function: t.Callable[[any], np.float32],
        fitness_compare_function: t.Callable[[any, any], bool],
        selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ],
        criteria_function: t.Callable[[np.uint64, t.List[DecodedIndividual]], bool],
        crossover_points: t.List[np.uint32],
        mutation_chance: np.float16,
        hillclimber_neighbor_selection_function: t.Union[None, t.Callable[[any], bool]],
        hillclimber_run_interval: np.uint32,
        hillclimber_step: np.float32 = np.float32(0.1),
        hillclimber_acceleration: np.float32 = np.float32(0.1),
        debug: bool = False,
    ) -> None:
        super().__init__(
            encode=encode,
            decode=decode,
            generate_initial_population=generate_initial_population,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            selection_function=selection_function,
            criteria_function=criteria_function,
            crossover_points=crossover_points,
            mutation_chance=mutation_chance,
            hillclimber_neighbor_selection_function=hillclimber_neighbor_selection_function,
            hillclimber_run_interval=hillclimber_run_interval,
            hillclimber_step=hillclimber_step,
            hillclimber_acceleration=hillclimber_acceleration,
            debug=debug,
        )
        self._metrics_runtime = []
        self._metrics_values = []
        self._metrics_fitness = []

    @staticmethod
    def from_function_definition(
        function_definition: FunctionDefinition,
        encode: t.Callable[[T], npt.NDArray[np.uint8]] = lambda x: np.array(
            [
                value
                for xi in x
                for value in np.frombuffer(
                    np.array([xi], dtype=np.float32).tobytes(), dtype=np.uint8
                )
            ]
        ),
        decode: t.Callable[[npt.NDArray[np.uint8]], T] = lambda x: np.array(
            [
                np.frombuffer(np.array(batch).tobytes(), dtype=np.float32)[0]
                for batch in [x[i : i + 4] for i in range(0, len(x), 4)]
            ]
        ),
        dimensions: int = 1,
        population_size: int = 100,
        generations: int = 100,
        selection_function: t.Union[
            t.Literal["auto"],
            t.Callable[
                [t.List[DecodedIndividual]],
                t.Tuple[DecodedIndividual, DecodedIndividual],
            ],
        ] = "auto",
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[t.List[np.float32], np.float32, np.uint64], bool],
        ] = "auto",
        crossover_points: t.List[np.uint32] = [],
        mutation_chance: np.float16 = 0.02,
        hillclimber_neighbor_selection_function: t.Union[
            None, t.Callable[[T], bool]
        ] = None,
        hillclimber_run_interval: np.uint32 = np.uint32(10),
        hillclimber_step: np.float32 = np.float32(0.1),
        hillclimber_acceleration: np.float32 = np.float32(0.1),
        debug: bool = False,
    ):
        cached_min_best_result = function_definition.best_result - 0.05
        cached_max_best_result = function_definition.best_result + 0.05
        selection_function = (
            selection_function
            if selection_function != "auto"
            else roulette_wheel_selection(target=function_definition.target)
        )
        criteria_function = (
            criteria_function
            if criteria_function != "auto"
            else lambda _values, fitness, generation: generation > generations
            or cached_min_best_result < fitness < cached_max_best_result
        )

        return MeteredHybridAlgorithm(
            encode=encode,
            decode=decode,
            generate_initial_population=lambda: [
                np.array([
                    np.random.uniform(
                        low=function_definition.value_boundaries.min,
                        high=function_definition.value_boundaries.max,
                    )
                    for _ in range(dimensions)
                ])
                for _ in range(population_size)
            ],
            fitness_function=function_definition.function,
            fitness_compare_function=maximise
            if function_definition.target == "maximise"
            else minimise,
            selection_function=selection_function,
            criteria_function=criteria_function,
            crossover_points=crossover_points,
            mutation_chance=mutation_chance,
            hillclimber_neighbor_selection_function=hillclimber_neighbor_selection_function,
            hillclimber_run_interval=hillclimber_run_interval,
            hillclimber_step=hillclimber_step,
            hillclimber_acceleration=hillclimber_acceleration,
            debug=debug,
        )

    def run(self) -> t.Tuple[any, any, np.uint64]:
        then = time.time_ns()
        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )
        best_individual = self._population[0]

        while not self._criteria_function(
            best_individual.fitness, best_individual.value, self._generation
        ):
            self.step()

            now = time.time_ns()
            self._metrics_runtime.append((self._generation, now - then))

            for individual in self.decoded_population:
                self._metrics_values.append((self._generation, individual[0]))
                self._metrics_fitness.append((self._generation, individual[1]))

        best_individual = self._population[0]

        return best_individual.fitness, best_individual.value, self._generation

    @property
    def metrics_runtime(self) -> t.List[t.Tuple[np.uint64, np.uint64]]:
        return self._metrics_runtime

    @property
    def metrics_values(self) -> t.List[t.Tuple[np.uint64, any]]:
        return self._metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_fitness
