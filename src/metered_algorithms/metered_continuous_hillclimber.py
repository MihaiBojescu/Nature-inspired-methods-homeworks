import typing as t
import numpy as np
from algorithms.continuous_hillclimber import ContinuousHillclimber


class MeteredContinuousHillclimber(ContinuousHillclimber):
    _metrics: t.List[t.Tuple[np.uint64, np.float32]]

    def __init__(
        self,
        fx: t.Callable[[np.float32], np.float32],
        initial_x: t.Union[t.Tuple[np.float32, np.float32], np.float32],
        step: np.float32,
        acceleration: np.float32,
        precision: np.float32 = np.finfo(np.float32).eps,
        generations: t.Union[None, np.uint64] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            fx=fx,
            initial_x=initial_x,
            step=step,
            acceleration=acceleration,
            precision=precision,
            generations=generations,
            debug=debug,
        )
        self._metrics = []

    def run(self) -> t.Tuple[np.float32, np.uint64]:
        while (
            (self._before_best_score is None)
            or (
                self._generations is None
                and np.abs(self._best_score - self._before_best_score) > self._precision
            )
            or (
                self._generations is not None
                and (
                    self._generation < self._generations
                    and np.abs(self._best_score - self._before_best_score)
                    > self._precision
                )
            )
        ):
            self.step()
            self._metrics.append((self._generation, self._best_x))

        return self._best_x, self._generation

    @property
    def metrics(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self._metrics
