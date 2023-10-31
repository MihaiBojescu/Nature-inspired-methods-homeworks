import random
import typing as t
import numpy as np
import numpy.typing as npt
from individual import DecodedIndividual, Individual


class BinaryGeneticAlgorithm:
    _population: t.List[Individual]
    _encode: t.Callable[[any], npt.NDArray[np.uint8]]
    _decode: t.Callable[[npt.NDArray[np.uint8]], any]
    _fitness_function: t.Callable[[any], np.float32]
    _criteria_function: t.Callable[[np.uint64, t.List[DecodedIndividual]], bool]
    _selection_function: t.Callable[
        [t.List[DecodedIndividual]],
        t.Tuple[DecodedIndividual, DecodedIndividual],
    ]
    _crossover_bits: t.List[np.uint8]
    _crossover_bytes: t.List[np.uint32]
    _mutation_chance: np.float16
    _debug: bool

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
        self._population = [
            Individual(genes, encode, decode, fitness_function)
            for genes in generate_initial_population()
        ]
        self._encode = encode
        self._decode = decode
        self._fitness_function = fitness_function
        self._criteria_function = criteria_function
        self._selection_function = selection_function
        self._crossover_bits = [
            np.uint8(crossover_point % 8) for crossover_point in crossover_points
        ]
        self._crossover_bytes = [
            np.uint8(crossover_point // 8) for crossover_point in crossover_points
        ]
        self._mutation_chance = mutation_chance
        self._debug = debug

    def run(self) -> t.Tuple[any, np.uint64, t.List[Individual]]:
        generation = np.uint64(0)

        while not self._criteria_function(generation, self.decoded_population):
            self._print(generation)
            self._population.sort(
                key=lambda individual: individual.fitness, reverse=True
            )
            self._population = self.step(self._population)
            generation += 1

        self._population.sort(key=lambda individual: individual.fitness, reverse=True)

        return self._population[0].decode()[0], generation, self._population

    def step(self, population: t.List[Individual]):
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

            child_1, child_2 = self._crossover(parent_1, parent_2)

            child_1 = self._mutate(child_1)
            child_2 = self._mutate(child_2)

            next_generation.extend([child_1, child_2])

        return next_generation

    @property
    def population(self) -> t.List[Individual]:
        return self._population

    @property
    def decoded_population(self) -> t.List[Individual]:
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
