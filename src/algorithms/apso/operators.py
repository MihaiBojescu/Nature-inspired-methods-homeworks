import typing as t
import numpy as np


class TwoOptOperator:
    __fitness_function: t.Callable[[t.List[np.int64]], np.float32]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[t.List[np.int64]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, values: t.List[np.uint64]):
        values_copy = values.copy()
        improved = True

        while improved:
            improved = False

            for i in range(1, len(values_copy) - 2):
                for j in range(i + 1, len(values_copy) - 1):
                    a = values_copy[i - 1]
                    b = values_copy[i]
                    c = values_copy[j]
                    d = values_copy[j + 1]

                    if self.__is_cost_lower_after_swap(a, b, c, d):
                        values_copy[b:d] = reversed(values_copy[b:d])
                        improved = True

        return values_copy

    def __is_cost_lower_after_swap(self, a: np.int64, b: np.int64, c: np.int64, d: np.int64):
        original_cost = self.__fitness_function([a, b]) + self.__fitness_function([c, d])
        swapped_cost = self.__fitness_function([a, c]) + self.__fitness_function([b, d])

        return self.__fitness_compare_function(original_cost, swapped_cost)


class PathLinkerOperator:
    __fitness_function: t.Callable[[t.List[np.int64]], np.float32]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[t.List[np.int64]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, values: t.List[np.int64]):
        values_copy = values.copy()
        improved = True

        while improved:
            improved = False

            for i in range(len(values_copy) - 1):
                for j in range(i + 1, len(values_copy)):
                    swapped_values = self.__swap(values_copy, i, j)

                    if self.__compare(values_copy, swapped_values):
                        values_copy = swapped_values
                        improved = True

        return values_copy

    def __swap(self, values: t.List[np.int64], i: int, j: int):
        swapped_values = values.copy()
        swapped_values[i], swapped_values[j] = swapped_values[j], swapped_values[i]

        return swapped_values

    def __compare(self, original_values: t.List[np.int64], swapped_values: t.List[np.int64]):
        original_cost = self.__fitness_function(original_values)
        swapped_cost = self.__fitness_function(swapped_values)

        return self.__fitness_compare_function(original_cost, swapped_cost)


class SwapOperator:
    def run(self, values: t.List[np.int64]):
        values_copy = values.copy()

        [a, b] = np.random.choice(list(range(len(values_copy))), size=2, replace=False)
        values_copy[a], values_copy[b] = values_copy[b], values_copy[a]

        return values_copy
