import time
import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.binary_hillclimber.binary_hillclimber import BinaryHillclimber
from functions.definition import FunctionDefinition
from util.sort import maximise, minimise

T = t.TypeVar("T")


class MeteredBinaryHillclimber(BinaryHillclimber):
    _metrics_runtime: t.List[t.Tuple[np.uint64, np.uint64]]
    _metrics_values: t.List[t.Tuple[np.uint64, np.float32]]
    _metrics_fitness: t.List[t.Tuple[np.uint64, np.float32]]

    def __init__(
        self,
        encode: t.Callable[[T], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], T],
        generate_initial_value: t.Callable[[], T],
        fitness_function: t.Callable[[T], np.float32],
        fitness_compare_function: t.Callable[[T, T], bool],
        neighbor_selection_function: t.Union[None, t.Callable[[T], bool]],
        criteria_function: t.Callable[[T, np.float32, np.uint64], bool],
        bit_shifts: np.uint32,
        debug: bool = False,
    ) -> None:
        super().__init__(
            encode=encode,
            decode=decode,
            generate_initial_value=generate_initial_value,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            neighbor_selection_function=neighbor_selection_function,
            criteria_function=criteria_function,
            bit_shifts=bit_shifts,
            debug=debug,
        )
        self._metrics_runtime = []
        self._metrics_values = []
        self._metrics_fitness = []

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
        bit_shifts: np.uint32 = np.uint32(1),
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

        return MeteredBinaryHillclimber(
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
            bit_shifts=bit_shifts,
            debug=debug,
        )

    def run(self) -> t.Tuple[T, np.float32, np.uint64]:
        then = time.time_ns()

        while not self._criteria_function(
            self._decode(self._best_value), self._best_score, self._generation
        ):
            self.step()

            now = time.time_ns()
            self._metrics_runtime.append((self._generation, now - then))
            self._metrics_values.append((self._generation, self._best_value))
            self._metrics_fitness.append((self._generation, self._best_score))

        return self._best_score, self._decode(self._best_value), self._generation

    @property
    def metrics_runtime(self) -> t.List[t.Tuple[np.uint64, np.uint64]]:
        return self._metrics_runtime

    @property
    def metrics_values(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_fitness
