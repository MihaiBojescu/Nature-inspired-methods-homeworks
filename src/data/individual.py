import typing as t
import numpy as np
import numpy.typing as npt

DecodedIndividual = t.Tuple[any, np.float32]

class Individual:
    _genes: npt.NDArray[np.uint8]
    _fitness: np.float32
    _encode: t.Callable[[any], npt.NDArray[np.uint8]]
    _decode: t.Callable[[npt.NDArray[np.uint8]], any]
    _fitness_function: t.Callable[[any], np.float32]

    def __init__(
        self,
        genes: npt.NDArray[np.uint8],
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        fitness_function: t.Callable[[any], np.float32],
    ) -> None:
        self._genes = encode(genes)
        self._fitness = fitness_function(genes)
        self._encode = encode
        self._decode = decode
        self._fitness_function = fitness_function

    @staticmethod
    def from_encoded_genes(
        genes: npt.NDArray[np.uint8],
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        fitness_function: t.Callable[[any], np.float32],
    ):
        return Individual(decode(genes), encode, decode, fitness_function)

    @staticmethod
    def from_decoded_individual(
        decoded_individual: DecodedIndividual,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        fitness_function: t.Callable[[any], np.float32],
    ):
        return Individual(decoded_individual[0], encode, decode, fitness_function)

    def encode(self) -> t.Tuple[npt.NDArray[np.uint8], np.float32]:
        return (self._genes, self._fitness)

    def decode(self) -> t.Tuple[any, np.float32]:
        return (self._decode(self._genes), self._fitness)

    @property
    def genes(self) -> npt.NDArray[np.uint8]:
        return self._genes

    @genes.setter
    def genes(self, genes: npt.NDArray[np.uint8]):
        self._genes = genes
        self._fitness = self._fitness_function(self._decode(self._genes))

    @property
    def fitness(self) -> np.float32:
        return self._fitness
