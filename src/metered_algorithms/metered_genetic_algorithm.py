import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.genetic_algorithm import BinaryGeneticAlgorithm
from data.individual import DecodedIndividual, Individual


class MeteredBinaryGenericAlgorithm(BinaryGeneticAlgorithm):
    _metrics_fitness: t.List[t.Tuple[np.uint64, t.List[np.float32]]]

    def __init__(
        self,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        generate_initial_population: t.Callable[[], t.List[any]],
        fitness_function: t.Callable[[any], np.float32],
        criteria_function: t.Callable[[np.uint64, t.List[DecodedIndividual]], bool],
        selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ],
        crossover_points: t.List[np.uint32],
        mutation_chance: np.float16,
        debug: bool = False,
    ) -> None:
        super().__init__(
            encode=encode,
            decode=decode,
            generate_initial_population=generate_initial_population,
            fitness_function=fitness_function,
            criteria_function=criteria_function,
            selection_function=selection_function,
            crossover_points=crossover_points,
            mutation_chance=mutation_chance,
            debug=debug,
        )
        self._metrics_fitness = []

    def run(self) -> t.Tuple[any, np.uint64, t.List[Individual]]:
        self._population.sort(key=lambda individual: individual.fitness, reverse=True)

        while not self._criteria_function(self._generation, self.decoded_population):
            self.step()
            self._metrics_fitness.append(
                (
                    self._generation,
                    [individual[1] for individual in self.decoded_population],
                )
            )

        return self._population[0].decode()[0], self._generation, self._population

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[np.uint64, t.List[np.float32]]]:
        return self._metrics_fitness
