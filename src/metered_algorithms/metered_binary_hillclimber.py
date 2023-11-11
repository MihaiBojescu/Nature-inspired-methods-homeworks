import time
import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.binary_hillclimber import BinaryHillclimber


class MeteredBinaryHillclimber(BinaryHillclimber):
    _metrics_runtime: t.List[t.Tuple[np.uint64, np.uint64]]
    _metrics_best_value: t.List[t.Tuple[np.uint64, np.float32]]
    _metrics_best_score: t.List[t.Tuple[np.uint64, np.float32]]

    def __init__(
        self,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        generate_initial_value: t.Callable[[], any],
        fitness_function: t.Callable[[any], np.float32],
        fitness_compare_function: t.Callable[[any, any], bool],
        neighbor_selection_function: t.Union[None, t.Callable[[any], bool]],
        criteria_function: t.Callable[[any, any, np.uint64], bool],
        debug: bool = False,
    ) -> None:
        super().__init__(
            generate_initial_value=generate_initial_value,
            encode=encode,
            decode=decode,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            neighbor_selection_function=neighbor_selection_function,
            criteria_function=criteria_function,
            debug=debug,
        )
        self._metrics_runtime = []
        self._metrics_best_value = []
        self._metrics_best_step = []
        self._metrics_best_score = []

    def run(self) -> t.Tuple[np.float32, np.uint64]:
        then = time.time_ns()

        while not self._criteria_function(
            self._best_score, self._decode(self._best_value), self._generation
        ):
            self.step()

            now = time.time_ns()
            self._metrics_runtime.append((self._generation, now - then))
            self._metrics_best_value.append((self._generation, self._best_value))
            self._metrics_best_step.append((self._generation, self._best_step))
            self._metrics_best_score.append((self._generation, self._best_score))

        return self._best_value, self._generation

    @property
    def metrics_runtime(self) -> t.List[t.Tuple[np.uint64, np.uint64]]:
        return self._metrics_runtime

    @property
    def metrics_best_value(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_best_value

    @property
    def metrics_best_score(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_best_score
