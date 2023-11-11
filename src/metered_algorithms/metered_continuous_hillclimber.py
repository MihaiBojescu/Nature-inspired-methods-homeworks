import time
import typing as t
import numpy as np
from algorithms.continuous_hillclimber import ContinuousHillclimber


class MeteredContinuousHillclimber(ContinuousHillclimber):
    _metrics_runtime: t.List[t.Tuple[np.uint64, np.uint64]]
    _metrics_best_value: t.List[t.Tuple[np.uint64, np.float32]]
    _metrics_best_step: t.List[t.Tuple[np.uint64, np.float32]]
    _metrics_best_score: t.List[t.Tuple[np.uint64, np.float32]]

    def __init__(
        self,
        generate_initial_value: t.Callable[[], np.float32],
        fitness_function: t.Callable[[np.float32], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        neighbor_selection_function: t.Union[None, t.Callable[[np.float32], bool]],
        criteria_function: t.Callable[[np.float32, np.float32, np.uint64], bool],
        step: np.float32,
        acceleration: np.float32,
        debug: bool = False,
    ) -> None:
        super().__init__(
            generate_initial_value=generate_initial_value,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            neighbor_selection_function=neighbor_selection_function,
            criteria_function=criteria_function,
            step=step,
            acceleration=acceleration,
            debug=debug,
        )
        self._metrics_runtime = []
        self._metrics_best_value = []
        self._metrics_best_step = []
        self._metrics_best_score = []

    def run(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        then = time.time_ns()

        while not self._criteria_function(
            self._best_score, self._best_value, self._generation
        ):
            self.step()

            now = time.time_ns()
            self._metrics_runtime.append((self._generation, now - then))
            self._metrics_best_value.append((self._generation, self._best_value))
            self._metrics_best_step.append((self._generation, self._best_step))
            self._metrics_best_score.append((self._generation, self._best_score))

        return self._best_score, self._best_value, self._generation

    @property
    def metrics_runtime(self) -> t.List[t.Tuple[np.uint64, np.uint64]]:
        return self._metrics_runtime

    @property
    def metrics_best_value(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_best_value

    @property
    def metrics_best_step(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_best_step

    @property
    def metrics_best_score(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics_best_score
