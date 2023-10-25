import typing as t
import numpy as np
import numpy.typing as npt


class ContinuousHillclimber:
    _fx: t.Callable[[np.float32], np.float32]
    _interval: t.Tuple[np.float32, np.float32]
    _step: np.float32
    _acceleration: np.float32
    _precision: np.float32
    _generations: t.Union[None, np.uint64]
    _step_candidates: npt.NDArray[np.float32]
    _debug: bool

    def __init__(
        self,
        fx: t.Callable[[np.float32], np.float32],
        interval: t.Tuple[np.float32, np.float32],
        step: np.float32,
        acceleration: np.float32,
        precision: np.float32 = np.finfo(np.float32).eps,
        generations: t.Union[None, np.uint64] = None,
        debug: bool = False,
    ) -> None:
        self._fx = fx
        self._interval = interval
        self._step = step
        self._acceleration = acceleration
        self._precision = precision
        self._generations = generations
        self._step_candidates = np.array(
            [
                self._acceleration,
                -self._acceleration,
                1 / self._acceleration,
                -1 / self._acceleration,
            ]
        )
        self._debug = debug

    def run(self, initial_x: t.Union[None, np.float32] = None):
        generation = 0
        best_step = self._step
        best_x = (
            np.random.uniform(self._interval[0], self._interval[1])
            if initial_x is None
            else initial_x
        )
        best_score = self._fx(best_x)
        before_score = None

        while (
            (before_score is None)
            or (
                self._generations is None
                and np.abs(best_score - before_score) > self._precision
            )
            or (
                self._generations is not None
                and (
                    generation < self._generations
                    and np.abs(best_score - before_score) > self._precision
                )
            )
        ):
            self._print(generation)
            before_score = best_score

            for step_candidate in self._step_candidates:
                step = best_step * step_candidate
                x = best_x + step
                score = self._fx(x)

                if score > best_score:
                    best_step = step
                    best_x = x
                    best_score = score

            if not (self._interval[0] < best_x < self._interval[1]):
                best_step = self._step
                best_x = (
                    self._interval[0]
                    if best_x > self._interval[1]
                    else self._interval[1]
                )
                best_score = self._fx(best_x)

            generation += 1

        return best_x

    def _print(self, generation: np.uint64) -> None:
        if self._debug:
            print(f"Continuous hillclimber algorithm generation: {generation}")
