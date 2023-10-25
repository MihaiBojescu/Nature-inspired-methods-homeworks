
import typing as t
import numpy as np
import numpy.typing as npt
from individual import DecodedIndividual, Individual
from genetic_algorithm import BinaryGeneticAlgorithm


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
        debug: bool = False,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self._fitness_function = fitness_function
        self._selection_function = selection_function
        self._crossover_point = crossover_point
        self._crossover_bit = np.uint8(self._crossover_point % 8)
        self._crossover_byte = np.uint32(self._crossover_point // 8)
        self._mutation_chance = mutation_chance
        self._debug = debug

    def run(self, population: list[Individual]) -> any:
        self._print()
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

    def _print(self) -> None:
        if self._debug:
            print(f"One-shot genetic algorithm")

