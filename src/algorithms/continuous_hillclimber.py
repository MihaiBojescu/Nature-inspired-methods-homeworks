import typing as t
import numpy as np
import numpy.typing as npt


class ContinuousHillclimber:
    _fx: t.Callable[[np.float32], np.float32]
    _step: np.float32
    _acceleration: np.float32
    _precision: np.float32
    _generations: t.Union[None, np.uint64]
    _step_candidates: npt.NDArray[np.float32]

    _best_x: np.float32
    _best_step: np.float32
    _best_score: np.float32
    _before_best_score: np.float32

    _debug: bool

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
        self._fx = fx
        self._step = step
        self._best_step = step
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

        self._generation = 0
        self._best_step = self._step
        self._best_x = (
            np.random.uniform(initial_x[0], initial_x[1])
            if type(initial_x) is tuple
            else initial_x
        )
        self._best_score = self._fx(self._best_x)
        self._before_best_score = None

        self._debug = debug

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

        return self._best_x, self._generation

    def step(self) -> t.Tuple[np.float32, np.float32]:
        self._print(self._generation)
        self._before_best_score = self._best_score

        for step_candidate in self._step_candidates:
            step = self._best_step * step_candidate
            x = self._best_x + step
            score = self._fx(x)

            if score > self._best_score:
                self._best_step = step
                self._best_x = x
                self._best_score = score

        self._generation += 1

        return self._best_x, self._generation

    def _print(self, generation: np.uint64) -> None:
        if self._debug:
            print(f"Continuous hillclimber algorithm generation: {generation}")
