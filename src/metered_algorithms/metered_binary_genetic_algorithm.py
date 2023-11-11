import time
import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.binary_genetic_algorithm import BinaryGeneticAlgorithm
from data.individual import DecodedIndividual, Individual


class MeteredBinaryGenericAlgorithm(BinaryGeneticAlgorithm):
    _metrics_runtime: t.List[t.Tuple[np.uint64, np.uint64]]
    _metrics_values: t.List[t.Tuple[np.uint64, any]]
    _metrics_fitness: t.List[t.Tuple[np.uint64, np.float32]]

    def __init__(
        self,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        generate_initial_population: t.Callable[[], t.List[any]],
        fitness_compare_function: t.Callable[[any, any], bool],
        fitness_function: t.Callable[[any], np.float32],
        selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ],
        criteria_function: t.Callable[[np.uint64, t.List[DecodedIndividual]], bool],
        crossover_points: t.List[np.uint32],
        mutation_chance: np.float16,
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
            debug=debug,
        )
        self._metrics_runtime = []
        self._metrics_values = []
        self._metrics_fitness = []

    def run(self) -> t.Tuple[any, any, np.uint64]:
        then = time.time_ns()
        self._population.sort(key=lambda individual: individual.fitness, reverse=True)

        while not self._criteria_function(self._generation, self.decoded_population):
            self.step()

            now = time.time_ns()
            self._metrics_runtime.append((self._generation, now - then))

            for individual in self.decoded_population:
                self._metrics_values.append((self._generation, individual[0]))
                self._metrics_fitness.append((self._generation, individual[1]))

        best_individual_decoded = self._population[0].decode()

        return best_individual_decoded[1], best_individual_decoded[0], self._generation

    @property
    def metrics_runtime(self) -> t.List[t.Tuple[np.uint64, np.uint64]]:
        return self._metrics_runtime

    @property
    def metrics_values(self) -> t.List[t.Tuple[np.uint64, any]]:
        return self._metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_fitness
