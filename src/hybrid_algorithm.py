import typing as t
import numpy as np
import numpy.typing as npt
from continuous_hillclimber import ContinuousHillclimber
from genetic_algorithm import BinaryGeneticAlgorithm
from individual import DecodedIndividual, Individual
from one_shot_genetic_algorithm import OneShotBinaryGeneticAlgorithm


class HybridAlgorithm:
    _generations: np.uint64
    _genetic_algorithm_encode: t.Callable[[any], npt.NDArray[np.uint8]]
    _hillclimber_run_interval: int
    _genetic_algorithm: OneShotBinaryGeneticAlgorithm
    _hillclimber_algorithm: ContinuousHillclimber
    _fx: t.Callable[[np.float32], np.float32]
    _debug: bool

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[any]],
        generations: np.uint64,
        genetic_algorithm_encode: t.Callable[[any], npt.NDArray[np.uint8]],
        genetic_algorithm_decode: t.Callable[[npt.NDArray[np.uint8]], any],
        genetic_algorithm_selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ],
        genetic_algorithm_crossover_points: t.List[np.uint32],
        genetic_algorithm_mutation_chance: np.float16,
        hillclimber_run_times: int,
        hillclimber_interval: t.Tuple[np.float32, np.float32],
        hillclimber_step: np.float32,
        hillclimber_acceleration: np.float32,
        hillclimber_precision: np.float32,
        hillclimber_generations: t.Union[None, np.uint64],
        fx: t.Callable[[any], np.float32],
        debug: bool = False,
    ) -> None:
        self._generations = generations
        self._genetic_algorithm_encode = genetic_algorithm_encode
        self._hillclimber_run_interval = (
            generations // hillclimber_run_times
            if generations > hillclimber_run_times
            else 1
        )
        self._fx = fx
        self._debug = debug

        self._genetic_algorithm = BinaryGeneticAlgorithm(
            generate_initial_population=generate_initial_population,
            encode=genetic_algorithm_encode,
            decode=genetic_algorithm_decode,
            fitness_function=fx,
            criteria_function=lambda: None,
            selection_function=genetic_algorithm_selection_function,
            crossover_points=genetic_algorithm_crossover_points,
            mutation_chance=genetic_algorithm_mutation_chance,
            debug=debug,
        )
        self._hillclimber_algorithm = ContinuousHillclimber(
            fx=fx,
            interval=hillclimber_interval,
            step=hillclimber_step,
            acceleration=hillclimber_acceleration,
            precision=hillclimber_precision,
            generations=hillclimber_generations,
            debug=debug,
        )

    def run(self):
        generation = np.uint64(0)

        while generation < self._generations:
            self._print(generation)

            self._run_genetic_algorithm()
            self._run_hillclimber(generation)

            generation += 1

        self._genetic_algorithm.population.sort(key=lambda individual: individual.fitness, reverse=True)

        return self._genetic_algorithm.population[0].decode()[0]

    def _print(self, generation: np.uint64) -> None:
        if self._debug:
            print(f"Hybrid algorithm generation: {generation}")

    def _run_genetic_algorithm(self):
        self._genetic_algorithm.step(self._genetic_algorithm.population)

    def _run_hillclimber(self, generation: int):
        if generation % self._hillclimber_run_interval != 0:
            return

        for i, individual in enumerate(self._genetic_algorithm.population):
            decoded_individual = individual.decode()[0]
            optimised_individual, _ = self._hillclimber_algorithm.run(decoded_individual)
            encoded_individual = self._genetic_algorithm_encode(optimised_individual)

            self._genetic_algorithm.population[i].genes = encoded_individual
