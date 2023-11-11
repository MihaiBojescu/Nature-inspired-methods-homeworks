import typing as t
import numpy as np
import numpy.typing as npt


class BinaryHillclimber:
    _encode: t.Callable[[any], npt.NDArray[np.uint8]]
    _decode: t.Callable[[npt.NDArray[np.uint8]], any]
    _fitness_function: t.Callable[[any], np.float32]
    _fitness_compare_function: t.Callable[[any, any], bool]
    _criteria_function: t.Callable[[any, any, np.uint64], bool]
    _neighbor_selection_function: t.Callable[[any], bool]

    _best_value: np.float32
    _best_score: np.float32
    _before_best_score: np.float32
    _value_bits: np.int32

    _debug: bool

    def __init__(
        self,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        generate_initial_value: t.Callable[[], any],
        fitness_function: t.Callable[[np.float32], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        criteria_function: t.Callable[[np.float32, np.uint64], bool],
        neighbor_selection_function: t.Union[None, t.Callable[[any], bool]],
        debug: bool = False,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self._fitness_function = fitness_function
        self._fitness_compare_function = fitness_compare_function
        self._criteria_function = criteria_function
        self._neighbor_selection_function = (
            neighbor_selection_function
            if neighbor_selection_function is not None
            else lambda _: True
        )

        self._generation = 0
        self._best_value = self._encode(generate_initial_value())
        self._best_score = self._fitness_function(self._decode(self._best_value))
        self._before_best_score = None
        self._value_bits = len(self._best_value) * 8

        self._debug = debug

    def run(self) -> t.Tuple[any, any, np.uint64]:
        while not self._criteria_function(
            self._best_score, self._decode(self._best_value), self._generation
        ):
            self.step()

        return self._best_score, self._decode(self._best_value), self._generation

    def step(self) -> t.Tuple[any, any, np.uint64]:
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

        return self._best_value, self._decode(self._best_value), self._generation

    def _print(self, generation: np.uint64) -> None:
        if self._debug:
            print(f"Binary hillclimber algorithm generation: {generation}")
