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
        criteria_function: t.Callable[[np.int64, t.List[DecodedIndividual]], bool],
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
        self._criteria_function = criteria_function
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
    _iterations: np.int32
    _genetic_algorithm_encode: t.Callable[[any], npt.NDArray[np.uint8]]
    _genetic_algorithm: OneShotBinaryGeneticAlgorithm
    _hillclimber_algorithm: ContinuousHillclimber
    _fx: t.Callable[[np.float32], np.float32]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[any]],
        iterations: np.int32,

        genetic_algorithm_encode: t.Callable[[any], npt.NDArray[np.uint8]],
        genetic_algorithm_decode: t.Callable[[npt.NDArray[np.uint8]], any],
        genetic_algorithm_fitness_function: t.Callable[[any], np.float32],
        genetic_algorithm_criteria_function: t.Callable[
            [np.int64, t.List[DecodedIndividual]], bool
        ],
        genetic_algorithm_selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ],
        genetic_algorithm_crossover_point: np.uint32,
        genetic_algorithm_mutation_chance: np.float16,

        hillclimber_interval: t.Tuple[np.float32, np.float32],
        hillclimber_step: np.float32,
        hillclimber_acceleration: np.float32,
        hillclimber_precision: np.float32 = np.finfo(np.float32).eps,
        hillclimber_iterations: t.Union[None, np.int32] = None,
    ) -> None:
        self._population = [
            Individual(
                genes,
                genetic_algorithm_encode,
                genetic_algorithm_decode,
                genetic_algorithm_fitness_function,
            )
            for genes in generate_initial_population()
        ]
        self._iterations = iterations
        self._genetic_algorithm_encode = genetic_algorithm_encode
        self._fx = genetic_algorithm_fitness_function

        self._genetic_algorithm = OneShotBinaryGeneticAlgorithm(
            encode=genetic_algorithm_encode,
            decode=genetic_algorithm_decode,
            fitness_function=genetic_algorithm_fitness_function,
            criteria_function=genetic_algorithm_criteria_function,
            selection_function=genetic_algorithm_selection_function,
            crossover_point=genetic_algorithm_crossover_point,
            mutation_chance=genetic_algorithm_mutation_chance,
        )
        self._hillclimber_algorithm = ContinuousHillclimber(
            fx=genetic_algorithm_fitness_function,
            interval=hillclimber_interval,
            step=hillclimber_step,
            acceleration=hillclimber_acceleration,
            precision=hillclimber_precision,
            iterations=hillclimber_iterations,
        )

    def run(self):
        generations = np.int32(0)

        while generations < self._iterations:
            self._population = self._genetic_algorithm.run(self._population)
            print(f"Generation: {generations}")

            for i, individual in enumerate(self._population):
                optimised_individual = self._hillclimber_algorithm.run(
                    individual.decode()[0]
                )
                self._population[i].genes = self._genetic_algorithm_encode(
                    optimised_individual
                )

            generations += 1

        self._population.sort(key=lambda individual: individual.fitness, reverse=True)

        return self._population[0].decode()[0]
