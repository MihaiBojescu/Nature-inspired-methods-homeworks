import typing as t
import numpy as np
import numpy.typing as npt


class ContinuousHillclimber:
    fx: t.Callable[[np.float32], np.float32]
    interval: t.Tuple[np.float32, np.float32]
    step: np.float32
    acceleration: np.float32
    precision: np.float32
    iterations: t.Union[None, np.int32]
    step_candidates: npt.NDArray[np.float32]

    def __init__(
        self,
        fx: t.Callable[[np.float32], np.float32],
        interval: t.Tuple[np.float32, np.float32],
        step: np.float32,
        acceleration: np.float32,
        precision: np.float32 = np.finfo(np.float32).eps,
        iterations: t.Union[None, np.int32] = None,
    ) -> None:
        self.fx = fx
        self.interval = interval
        self.step = step
        self.acceleration = acceleration
        self.precision = precision
        self.iterations = iterations
        self.step_candidates = np.array(
            [
                self.acceleration,
                -self.acceleration,
                1 / self.acceleration,
                -1 / self.acceleration,
            ]
        )

    def run(self):
        ran_iterations = 0
        best_step = self.step
        best_x = np.random.uniform(self.interval[0], self.interval[1])
        best_score = self.fx(best_x)
        before_score = None

        while (
            (self.iterations is not None and ran_iterations < self.iterations)
            or (before_score is None)
            or (np.abs(best_score - before_score) > self.precision)
        ):
            before_score = best_score

            for step_candidate in self.step_candidates:
                step = best_step * step_candidate
                x = best_x + step
                score = self.fx(x)

                if score > best_score:
                    best_step = step
                    best_x = x
                    best_score = score

            if not (self.interval[0] < best_x < self.interval[1]):
                best_step = self.step
                best_x = (
                    self.interval[0] if best_x > self.interval[1] else self.interval[1]
                )
                best_score = self.fx(best_x)

            ran_iterations += 1

        return best_x
