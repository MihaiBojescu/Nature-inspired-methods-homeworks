import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.continuous_hillclimber import ContinuousHillclimber
from algorithms.genetic_algorithm import BinaryGeneticAlgorithm
from data.individual import DecodedIndividual, Individual


class HybridAlgorithm(BinaryGeneticAlgorithm):
    _hillclimber_run_interval: np.uint32
    _hillclimber_step: np.float32
    _hillclimber_acceleration: np.float32
    _hillclimber_precision: np.float32

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
            debug=debug,
        )

        self._hillclimber_run_interval = hillclimber_run_interval
        self._hillclimber_step = hillclimber_step
        self._hillclimber_acceleration = hillclimber_acceleration
        self._hillclimber_precision = hillclimber_precision

    def step(self):
        self._print(self._generation)
        self._population.sort(key=lambda individual: individual.fitness, reverse=True)
        next_generation = []

        for _ in range(0, len(self._population) // 2):
            parent_1, parent_2 = self._selection_function(self.decoded_population)
            parent_1 = Individual.from_decoded_individual(
                parent_1, self._encode, self._decode, self._fitness_function
            )
            parent_2 = Individual.from_decoded_individual(
                parent_2, self._encode, self._decode, self._fitness_function
            )

            child_1, child_2 = self._crossover(parent_1, parent_2)

            child_1 = self._mutate(child_1)
            child_2 = self._mutate(child_2)

            next_generation.extend([child_1, child_2])

        self._population = next_generation
        self._run_hillclimber()
        self._population.sort(key=lambda individual: individual.fitness, reverse=True)
        self._generation += 1

        return self._population

    def _print(self, generation: np.uint64) -> None:
        if self._debug:
            print(f"Hybrid algorithm generation: {generation}")

    def _run_hillclimber(self):
        if self._generation % self._hillclimber_run_interval != 0:
            return

        for i, individual in enumerate(self._population):
            decoded_individual = individual.decode()[0]
            hill_climber = ContinuousHillclimber(
                fx=self._fitness_function,
                initial_x=decoded_individual,
                step=self._hillclimber_step,
                acceleration=self._hillclimber_acceleration,
                precision=self._hillclimber_precision,
                generations=-1,
                debug=self._debug,
            )
            optimised_individual, _ = hill_climber.step()
            encoded_individual = self._encode(optimised_individual)

            self._population[i].genes = encoded_individual
