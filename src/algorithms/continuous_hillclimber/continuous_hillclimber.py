import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.base.algorithm import BaseAlgorithm
from functions.definition import FunctionDefinition
from util.sort import maximise, minimise


class ContinuousHillclimber(BaseAlgorithm):
    _fitness_function: t.Callable[[np.float32], np.float32]
    _fitness_compare_function: t.Callable[[np.float32, np.float32], bool]
    _neighbor_selection_function: t.Callable[[np.float32], bool]
    _criteria_function: t.Callable[[np.float32, np.float32, np.int64], bool]

    _generation: np.uint64
    _step: np.float32
    _acceleration: np.float32
    _step_candidates: npt.NDArray[np.float32]
    _best_value: np.float32
    _best_step: np.float32
    _best_score: np.float32

    _debug: bool

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
        self._fitness_function = fitness_function
        self._fitness_compare_function = fitness_compare_function
        self._neighbor_selection_function = (
            neighbor_selection_function
            if neighbor_selection_function is not None
            else lambda _: True
        )
        self._criteria_function = criteria_function

        self._generation = np.uint64(0)
        self._step = step
        self._best_step = step
        self._acceleration = acceleration
        self._step_candidates = np.array(
            [
                self._acceleration,
                -self._acceleration,
                1 / self._acceleration,
                -1 / self._acceleration,
            ]
        )
        self._best_step = self._step
        self._best_value = generate_initial_value()
        self._best_score = self._fitness_function(self._best_value)

        self._debug = debug

    @staticmethod
    def from_function_definition(
        function_definition: FunctionDefinition,
        dimensions: int = 1,
        generations: int = 100,
        neighbor_selection_function: t.Union[
            None, t.Callable[[np.float32], bool]
        ] = lambda _: True,
        step: np.float32 = 0.01,
        acceleration: np.float32 = 0.01,
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

        return ContinuousHillclimber(
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
            step=step,
            acceleration=acceleration,
            debug=debug,
        )

    @property
    def name(self) -> str:
        return "Continuous hillclimber algorithm"

    def run(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        while not self._criteria_function(
            self._best_value, self._best_score, self._generation
        ):
            self.step()

        return self._best_value, self._best_score, self._generation

    def step(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        self._print(self._generation)

        for step_candidate in self._step_candidates:
            step = self._best_step * step_candidate
            value = self._best_value + step
            score = self._fitness_function(value)

            if not self._neighbor_selection_function(value):
                continue

            if self._fitness_compare_function(score, self._best_score):
                self._best_step = step
                self._best_value = value
                self._best_score = score

        self._generation += 1

        return self._best_value, self._best_score, self._generation

    def _print(self, generation: np.uint64) -> None:
        if self._debug:
            print(f"Continuous hillclimber algorithm generation: {generation}")
