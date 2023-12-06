import random
import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.base.algorithm import BaseAlgorithm
from algorithms.binary_genetic_algorithm.individual import DecodedIndividual, Individual
from algorithms.binary_genetic_algorithm.selection_functions import (
    roulette_wheel_selection,
)
from functions.definition import FunctionDefinition
from util.sort import maximise, minimise, quicksort

T = t.TypeVar("T")


class BinaryGeneticAlgorithm(BaseAlgorithm):
    _population: t.List[Individual]
    _encode: t.Callable[[T], npt.NDArray[np.uint8]]
    _decode: t.Callable[[npt.NDArray[np.uint8]], T]
    _fitness_function: t.Callable[[T], np.float32]
    _fitness_compare_function: t.Callable[[T, T], bool]
    _selection_function: t.Callable[
        [t.List[DecodedIndividual]],
        t.List[DecodedIndividual],
    ]
    _criteria_function: t.Callable[[T, T, np.uint64], bool]

    _crossover_bits: t.List[np.uint8]
    _crossover_bytes: t.List[np.uint32]
    _mutation_chance: np.float16
    _generation: np.uint64

    _debug: bool

    def __init__(
        self,
        encode: t.Callable[[T], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], T],
        generate_initial_population: t.Callable[[], t.List[T]],
        fitness_function: t.Callable[[T], np.float32],
        fitness_compare_function: t.Callable[[T, T], bool],
        selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.List[DecodedIndividual],
        ],
        criteria_function: t.Callable[[T, T, np.uint64], bool],
        crossover_points: t.List[np.uint32],
        mutation_chance: np.float16,
        debug: bool = False,
    ) -> None:
        self._population = [
            Individual(genes, encode, decode, fitness_function)
            for genes in generate_initial_population()
        ]
        self._encode = encode
        self._decode = decode
        self._fitness_function = fitness_function
        self._fitness_compare_function = fitness_compare_function
        self._selection_function = selection_function
        self._criteria_function = criteria_function

        self._crossover_bits = [
            np.uint8(crossover_point % 8) for crossover_point in crossover_points
        ]
        self._crossover_bytes = [
            np.uint8(crossover_point // 8) for crossover_point in crossover_points
        ]
        self._mutation_chance = mutation_chance
        self._generation = np.uint64(0)

        self._debug = debug

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
        decode: t.Callable[[npt.NDArray[np.uint8]], T] = lambda x: [
            np.frombuffer(np.array(batch).tobytes(), dtype=np.float32)[0]
            for batch in [x[i : i + 4] for i in range(0, len(x), 4)]
        ],
        dimensions: int = 1,
        population_size: int = 100,
        generations: int = 100,
        selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ] = roulette_wheel_selection,
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[t.List[np.float32], np.float32, np.uint64], bool],
        ] = "auto",
        crossover_points: t.List[np.uint32] = [],
        mutation_chance: np.float16 = 0.02,
        debug: bool = False,
    ):
        cached_min_best_result = function_definition.best_result - 0.05
        cached_max_best_result = function_definition.best_result + 0.05
        criteria_function = (
            criteria_function
            if criteria_function != "auto"
            else lambda _values, fitness, generation: generation > generations
            or cached_min_best_result < fitness < cached_max_best_result
        )

        return BinaryGeneticAlgorithm(
            encode=encode,
            decode=decode,
            generate_initial_population=lambda: [
                [
                    np.random.uniform(
                        low=function_definition.value_boundaries.min,
                        high=function_definition.value_boundaries.max,
                    )
                    for _ in range(dimensions)
                ]
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
            debug=debug,
        )

    def run(self) -> t.Tuple[T, np.float32, np.uint64]:
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

        best_individual = self._population[0]

        return best_individual.fitness, best_individual.value, self._generation

    def step(self) -> t.Tuple[T, np.float32, np.uint64]:
        self._print(self._generation)
        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )
        next_generation = []

        selected_population = self._selection_function(self.decoded_population)

        for index in range(0, len(selected_population) - 1, 2):
            parent_1 = Individual.from_decoded_individual(
                decoded_individual=selected_population[index],
                encode=self._encode,
                decode=self._decode,
                fitness_function=self._fitness_function,
            )
            parent_2 = Individual.from_decoded_individual(
                decoded_individual=selected_population[index + 1],
                encode=self._encode,
                decode=self._decode,
                fitness_function=self._fitness_function,
            )

            child_1, child_2 = self._crossover(parent_1, parent_2)

            child_1 = self._mutate(child_1)
            child_2 = self._mutate(child_2)

            next_generation.extend([child_1, child_2])

        self._population = next_generation
        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )
        self._generation += 1

        best_individual = self._population[0]

        return best_individual.fitness, best_individual.value, self._generation

    @property
    def population(self) -> t.List[Individual]:
        return self._population

    @property
    def decoded_population(self) -> t.List[t.Tuple[T, np.float32]]:
        return [individual.decode() for individual in self._population]

    def _print(self, generation: int) -> None:
        if self._debug:
            print(f"Genetic algorithm generation: {generation}")

    def _crossover(
        self, parent_1: Individual, parent_2: Individual
    ) -> t.Tuple[Individual, Individual]:
        p = np.random.uniform(low=0, high=1)

        previous_crossover_byte = 0
        parent_1_genes = parent_1.genes
        parent_2_genes = parent_2.genes
        genes_size = parent_1.genes.shape[0]

        child_1_genes = parent_1.genes.copy()
        child_2_genes = parent_2.genes.copy()

        for bit, byte in zip(self._crossover_bits, self._crossover_bytes):
            child_1_gene, child_2_gene = self._get_crossover_bytes(
                child_1_genes=child_1_genes, child_2_genes=child_2_genes, byte=byte
            )
            bit_mask_1, bit_mask_2 = self._get_crossover_bit_masks(bit=bit, p=p)
            rest_bytes_1, rest_bytes_2 = self._get_rest_bytes(
                parent_1_genes=parent_1_genes, parent_2_genes=parent_2_genes, p=p
            )
            crossovered_byte_1, crossovered_byte_2 = self._perform_crossover_on_bytes(
                child_1_gene=child_1_gene,
                child_2_gene=child_2_gene,
                bit_mask_1=bit_mask_1,
                bit_mask_2=bit_mask_2,
            )

            for i in range(previous_crossover_byte, byte - 1):
                child_1_genes[i] = rest_bytes_1[i]
                child_2_genes[i] = rest_bytes_2[i]

            child_1_genes[byte] = crossovered_byte_1
            child_2_genes[byte] = crossovered_byte_2

            previous_crossover_byte = byte
            p = 1 - p

        rest_bytes_1, rest_bytes_2 = self._get_rest_bytes(
            parent_1_genes=parent_1_genes, parent_2_genes=parent_2_genes, p=p
        )
        for i in range(previous_crossover_byte, genes_size):
            child_1_genes[i] = rest_bytes_1[i]
            child_2_genes[i] = rest_bytes_2[i]

        child_1 = Individual.from_encoded_genes(
            child_1_genes, self._encode, self._decode, self._fitness_function
        )
        child_2 = Individual.from_encoded_genes(
            child_2_genes, self._encode, self._decode, self._fitness_function
        )

        return child_1, child_2

    def _get_crossover_bytes(
        self,
        child_1_genes: npt.NDArray[np.uint8],
        child_2_genes: npt.NDArray[np.uint8],
        byte: np.uint32,
    ) -> t.Tuple[np.uint8, np.uint8]:
        child_1_gene = child_1_genes[byte]
        child_2_gene = child_2_genes[byte]

        return child_1_gene, child_2_gene

    def _get_crossover_bit_masks(
        self, bit: np.uint8, p: np.float32
    ) -> t.Tuple[np.uint8, np.uint8]:
        bit_mask_1 = 0
        bit_mask_2 = 0

        for i in range(0, bit):
            bit_mask_1 |= 1 << i

        for i in range(bit, 8):
            bit_mask_2 |= 1 << i

        if p > 0.5:
            return bit_mask_2, bit_mask_1

        return bit_mask_1, bit_mask_2

    def _perform_crossover_on_bytes(
        self,
        child_1_gene: np.uint8,
        child_2_gene: np.uint8,
        bit_mask_1: np.uint8,
        bit_mask_2: np.uint8,
    ) -> t.Tuple[np.uint8, np.uint8]:
        joint_gene_1 = (child_1_gene & bit_mask_1) | (child_2_gene & bit_mask_2)
        joint_gene_2 = (child_2_gene & bit_mask_1) | (child_1_gene & bit_mask_2)

        return joint_gene_1, joint_gene_2

    def _get_rest_bytes(
        self,
        parent_1_genes: npt.NDArray[np.uint8],
        parent_2_genes: npt.NDArray[np.uint8],
        p: np.float32,
    ) -> t.Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        if p > 0.5:
            return parent_2_genes, parent_1_genes

        return parent_1_genes, parent_2_genes

    def _mutate(self, child: Individual):
        child_genes = child.genes.copy()

        for position in range(0, child_genes.shape[0]):
            mask = np.uint8(0)

            for bit in range(0, 8):
                if random.random() > self._mutation_chance:
                    continue

                mask |= 1 << bit

            child_genes[position] ^= mask

        child.genes = child_genes

        return child
