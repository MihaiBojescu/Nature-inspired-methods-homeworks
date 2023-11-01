import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.hybrid_algorithm import HybridAlgorithm
from data.individual import DecodedIndividual, Individual


class MeteredHybridAlgorithm(HybridAlgorithm):
    _metrics: t.List[t.Tuple[t.uint64, t.List[np.float32]]]

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
        hillclimber_run_interval: np.uint32,
        hillclimber_step: np.float32 = np.float32(0.1),
        hillclimber_acceleration: np.float32 = np.float32(0.1),
        hillclimber_precision: np.float32 = np.finfo(np.float32).eps,
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
            hillclimber_run_interval=hillclimber_run_interval,
            hillclimber_step=hillclimber_step,
            hillclimber_acceleration=hillclimber_acceleration,
            hillclimber_precision=hillclimber_precision,
            debug=debug
        )
        self._metrics = []

    def run(self) -> t.Tuple[any, np.uint64, t.List[Individual]]:
        self._population.sort(key=lambda individual: individual.fitness, reverse=True)

        while not self._criteria_function(self._generation, self.decoded_population):
            self.step()
            self._metrics.append(
                (
                    self._generation,
                    [individual[1] for individual in self.decoded_population],
                )
            )

        return self._population[0].decode()[0], self._generation, self._population

    @property
    def metrics(self) -> t.List[t.Tuple[t.uint64, t.List[np.float32]]]:
        return self._metrics
