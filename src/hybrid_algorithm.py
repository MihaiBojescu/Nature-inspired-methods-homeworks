import typing as t
import numpy as np
import numpy.typing as npt
from continuous_hillclimber import ContinuousHillclimber

from genetic_algorithm import BinaryGeneticAlgorithm, DecodedIndividual, Individual


class OneShotBinaryGeneticAlgorithm(BinaryGeneticAlgorithm):
    def __init__(
        self,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        fitness_function: t.Callable[[any], np.float32],
        selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ],
        crossover_point: np.uint32,
        mutation_chance: np.float16,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self._fitness_function = fitness_function
        self._selection_function = selection_function
        self._crossover_point = crossover_point
        self._crossover_bit = np.uint8(self._crossover_point % 8)
        self._crossover_byte = np.uint32(self._crossover_point // 8)
        self._mutation_chance = mutation_chance

    def run(self, population: list[Individual]) -> any:
        population.sort(key=lambda individual: individual.fitness, reverse=True)
        next_generation = []

        for _ in range(0, len(population) // 2):
            parent_1, parent_2 = self._selection_function(
                [individual.decode() for individual in population]
            )
            parent_1 = Individual.from_decoded_individual(
                parent_1, self._encode, self._decode, self._fitness_function
            )
            parent_2 = Individual.from_decoded_individual(
                parent_2, self._encode, self._decode, self._fitness_function
            )

            child_1, child_2 = self._crossover_function(parent_1, parent_2)

            child_1 = self._mutate(child_1)
            child_2 = self._mutate(child_2)

            next_generation.extend([child_1, child_2])

        population = next_generation
        population.sort(key=lambda individual: individual.fitness, reverse=True)

        return population


class HybridAlgorithm:
    _population: list[Individual]
    _generations: np.int32
    _genetic_algorithm_encode: t.Callable[[any], npt.NDArray[np.uint8]]
    _hillclimber_run_interval: int
    _genetic_algorithm: OneShotBinaryGeneticAlgorithm
    _hillclimber_algorithm: ContinuousHillclimber
    _fx: t.Callable[[np.float32], np.float32]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[any]],
        generations: np.int32,
        genetic_algorithm_encode: t.Callable[[any], npt.NDArray[np.uint8]],
        genetic_algorithm_decode: t.Callable[[npt.NDArray[np.uint8]], any],
        genetic_algorithm_selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ],
        genetic_algorithm_crossover_point: np.uint32,
        genetic_algorithm_mutation_chance: np.float16,
        hillclimber_run_times: int,
        hillclimber_interval: t.Tuple[np.float32, np.float32],
        hillclimber_step: np.float32,
        hillclimber_acceleration: np.float32,
        hillclimber_precision: np.float32,
        fx: t.Callable[[any], np.float32],
    ) -> None:
        self._population = [
            Individual(
                genes,
                genetic_algorithm_encode,
                genetic_algorithm_decode,
                fx,
            )
            for genes in generate_initial_population()
        ]
        self._generations = generations
        self._genetic_algorithm_encode = genetic_algorithm_encode
        self._hillclimber_run_interval = (
            generations // hillclimber_run_times
            if generations > hillclimber_run_times
            else 1
        )
        self._fx = fx

        self._genetic_algorithm = OneShotBinaryGeneticAlgorithm(
            encode=genetic_algorithm_encode,
            decode=genetic_algorithm_decode,
            fitness_function=fx,
            selection_function=genetic_algorithm_selection_function,
            crossover_point=genetic_algorithm_crossover_point,
            mutation_chance=genetic_algorithm_mutation_chance,
        )
        self._hillclimber_algorithm = ContinuousHillclimber(
            fx=fx,
            interval=hillclimber_interval,
            step=hillclimber_step,
            acceleration=hillclimber_acceleration,
            precision=hillclimber_precision,
            iterations=1,
        )

    def run(self):
        generation = np.int32(0)

        while generation < self._generations:
            print(f"Hybrid algorithm generation: {generation}")

            self._run_genetic_algorithm()
            self._run_hillclimber(generation)

            generation += 1

        self._population.sort(key=lambda individual: individual.fitness, reverse=True)

        return self._population[0].decode()[0]

    def _run_genetic_algorithm(self):
        self._population = self._genetic_algorithm.run(self._population)

    def _run_hillclimber(self, generation: int):
        if generation % self._hillclimber_run_interval != 0:
            return

        for i, individual in enumerate(self._population):
            decoded_individual = individual.decode()[0]
            optimised_individual = self._hillclimber_algorithm.run(decoded_individual)
            encoded_individual = self._genetic_algorithm_encode(optimised_individual)

            self._population[i].genes = encoded_individual
