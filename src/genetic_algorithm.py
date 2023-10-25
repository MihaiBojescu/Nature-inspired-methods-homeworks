import random
import typing as t
import numpy as np
import numpy.typing as npt
from individual import DecodedIndividual, Individual


class BinaryGeneticAlgorithm:
    _population: list[Individual]
    _encode: t.Callable[[any], npt.NDArray[np.uint8]]
    _decode: t.Callable[[npt.NDArray[np.uint8]], any]
    _fitness_function: t.Callable[[any], np.float32]
    _criteria_function: t.Callable[[np.int64, t.List[DecodedIndividual]], bool]
    _selection_function: t.Callable[
        [t.List[DecodedIndividual]],
        t.Tuple[DecodedIndividual, DecodedIndividual],
    ]
    _crossover_point: np.uint32
    _crossover_bit: np.uint8
    _crossover_byte: np.uint32
    _mutation_chance: np.float16

    def __init__(
        self,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        generate_initial_population: t.Callable[[], t.List[any]],
        fitness_function: t.Callable[[any], np.float32],
        criteria_function: t.Callable[[np.int64, t.List[DecodedIndividual]], bool],
        selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.Tuple[DecodedIndividual, DecodedIndividual],
        ],
        crossover_point: np.uint32,
        mutation_chance: np.float16,
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
        self._crossover_point = crossover_point
        self._crossover_bit = np.uint8(self._crossover_point % 8)
        self._crossover_byte = np.uint32(self._crossover_point // 8)
        self._mutation_chance = mutation_chance

    def run(self) -> any:
        generations = np.int64(0)

        while not self._criteria_function(
            generations, [individual.decode() for individual in self._population]
        ):
            print(f"Genetic algorithm generation: {generations}")
            self._population.sort(
                key=lambda individual: individual.fitness, reverse=True
            )

            next_generation = []

            for _ in range(0, len(self._population) // 2):
                parent_1, parent_2 = self._selection_function(
                    [individual.decode() for individual in self._population]
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

            self._population = next_generation
            generations += 1

        self._population.sort(key=lambda individual: individual.fitness, reverse=True)

        return self._population[0].decode()[0]

    def _crossover_function(
        self, parent_1: Individual, parent_2: Individual
    ) -> t.Tuple[Individual, Individual]:
        parent_1_genes = parent_1.genes
        parent_2_genes = parent_2.genes

        crossover_gene_1, crossover_gene_2 = self._get_crossover_genes(
            parent_1_genes, parent_2_genes
        )
        bit_mask_1, bit_mask_2 = self._get_bit_masks()
        joint_gene_1, joint_gene_2 = self._get_joint_genes(
            crossover_gene_1, crossover_gene_2, bit_mask_1, bit_mask_2
        )

        child_1_genes = np.concatenate(
            (
                np.fromiter(
                    [parent_1_genes[i] for i in range(0, self._crossover_byte - 1)],
                    dtype=np.uint8,
                ),
                np.array([joint_gene_1], dtype=np.uint8),
                np.fromiter(
                    [
                        parent_2_genes[i]
                        for i in range(
                            self._crossover_byte + 1, parent_2_genes.shape[0]
                        )
                    ],
                    dtype=np.uint8,
                ),
            )
        )
        child_2_genes = np.concatenate(
            (
                np.fromiter(
                    [parent_2_genes[i] for i in range(0, self._crossover_byte - 1)],
                    dtype=np.uint8,
                ),
                np.array([joint_gene_2], dtype=np.uint8),
                np.fromiter(
                    [
                        parent_1_genes[i]
                        for i in range(
                            self._crossover_byte + 1, parent_1_genes.shape[0]
                        )
                    ],
                    dtype=np.uint8,
                ),
            )
        )

        child_1 = Individual.from_encoded_genes(
            child_1_genes, self._encode, self._decode, self._fitness_function
        )
        child_2 = Individual.from_encoded_genes(
            child_2_genes, self._encode, self._decode, self._fitness_function
        )

        return child_1, child_2

    def _get_crossover_genes(
        self,
        parent_1_genes: npt.NDArray[np.uint8],
        parent_2_genes: npt.NDArray[np.uint8],
    ) -> t.Tuple[np.uint8, np.uint8]:
        parent_1_crossover_gene = parent_1_genes[self._crossover_byte]
        parent_2_crossover_gene = parent_2_genes[self._crossover_byte]

        return parent_1_crossover_gene, parent_2_crossover_gene

    def _get_bit_masks(self) -> t.Tuple[np.uint8, np.uint8]:
        bit_mask_1 = 0
        bit_mask_2 = 0

        for i in range(0, self._crossover_bit):
            bit_mask_1 |= 1 << i

        for i in range(self._crossover_bit, 8):
            bit_mask_2 |= 1 << i

        return bit_mask_1, bit_mask_2

    def _get_joint_genes(
        self,
        crossover_gene_1: np.uint8,
        crossover_gene_2: np.uint8,
        bit_mask_1: np.uint8,
        bit_mask_2: np.uint8,
    ) -> t.Tuple[np.uint8, np.uint8]:
        joint_gene_1 = (crossover_gene_1 & bit_mask_1) | (crossover_gene_2 & bit_mask_2)
        joint_gene_2 = (crossover_gene_2 & bit_mask_1) | (crossover_gene_1 & bit_mask_2)

        return joint_gene_1, joint_gene_2

    def _mutate(self, child: Individual):
        child_genes = child.genes.copy()

        for position in range(0, child_genes.shape[0]):
            mask = np.uint8(0)

            for bit in range(0, 7):
                if random.random() > self._mutation_chance:
                    continue

                mask |= 1 << bit

            child_genes[position] ^= mask

        child.genes = child_genes

        return child
