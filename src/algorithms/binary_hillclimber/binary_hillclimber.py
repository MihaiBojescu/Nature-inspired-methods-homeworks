import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.base.algorithm import BaseAlgorithm
from functions.definition import FunctionDefinition
from util.sort import maximise, minimise

T = t.TypeVar("T")


class BinaryHillclimber(BaseAlgorithm):
    _encode: t.Callable[[T], npt.NDArray[np.uint8]]
    _decode: t.Callable[[npt.NDArray[np.uint8]], T]
    _fitness_function: t.Callable[[T], np.float32]
    _fitness_compare_function: t.Callable[[T, T], bool]
    _neighbor_selection_function: t.Callable[[T], bool]
    _criteria_function: t.Callable[[T, np.float32, np.uint64], bool]

    _generation: np.uint64
    _best_value: np.float32
    _best_score: np.float32
    _before_best_score: np.float32
    _value_bits: np.int32

    _debug: bool

    def __init__(
        self,
        encode: t.Callable[[T], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], T],
        generate_initial_value: t.Callable[[], T],
        fitness_function: t.Callable[[T], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        neighbor_selection_function: t.Union[None, t.Callable[[T], bool]],
        criteria_function: t.Callable[[T, np.float32, np.uint64], bool],
        debug: bool = False,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self._fitness_function = fitness_function
        self._fitness_compare_function = fitness_compare_function
        self._neighbor_selection_function = (
            neighbor_selection_function
            if neighbor_selection_function is not None
            else lambda _: True
        )
        self._criteria_function = criteria_function

        self._generation = np.uint64(0)
        self._best_value = self._encode(generate_initial_value())
        self._best_score = self._fitness_function(self._decode(self._best_value))
        self._before_best_score = None
        self._value_bits = len(self._best_value) * 8

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
        decode: t.Callable[[npt.NDArray[np.uint8]], T] = lambda x: np.array(
            [
                np.frombuffer(np.array(batch).tobytes(), dtype=np.float32)[0]
                for batch in [x[i : i + 4] for i in range(0, len(x), 4)]
            ]
        ),
        dimensions: int = 1,
        generations: int = 100,
        neighbor_selection_function: t.Union[
            None, t.Callable[[T], bool]
        ] = lambda _: True,
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[t.List[np.float32], np.float32, np.uint64], bool],
        ] = "auto",
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

        return BinaryHillclimber(
            encode=encode,
            decode=decode,
            generate_initial_value=lambda: np.array(
                [
                    np.random.uniform(
                        low=function_definition.value_boundaries.min,
                        high=function_definition.value_boundaries.max,
                    )
                    for _ in range(dimensions)
                ]
            ),
            fitness_function=function_definition.function,
            fitness_compare_function=maximise
            if function_definition.target == "maximise"
            else minimise,
            neighbor_selection_function=neighbor_selection_function,
            criteria_function=criteria_function,
            debug=debug,
        )

    @property
    def name(self) -> str:
        return "Binary hillclimber algorithm"

    def run(self) -> t.Tuple[T, np.float32, np.uint64]:
        while not self._criteria_function(
            self._decode(self._best_value), self._best_score, self._generation
        ):
            self.step()

        return self._decode(self._best_value), self._best_score, self._generation

    def step(self) -> t.Tuple[T, np.float32, np.uint64]:
        self._print(self._generation)
        self._before_best_score = self._best_score

        for _ in range(0, self._value_bits):
            random_bit = np.random.randint(low=0, high=self._value_bits)
            bit = random_bit % 8
            byte = random_bit // 8

            value = self._best_value.copy()
            value[byte] ^= 1 << bit
            value_decoded = self._decode(value)

            if not self._neighbor_selection_function(value_decoded):
                continue

            score = self._fitness_function(value_decoded)

            if self._fitness_compare_function(score, self._best_score):
                self._best_value = value
                self._best_score = score

        self._generation += 1

        return self._decode(self._best_value), self._best_value, self._generation

    def _print(self, generation: np.uint64) -> None:
        if self._debug:
            print(f"Binary hillclimber algorithm generation: {generation}")
