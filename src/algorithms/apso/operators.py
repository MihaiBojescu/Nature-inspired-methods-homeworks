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
        improved = True

        while improved:
            improved = False

            for i in range(1, len(values) - 2):
                for j in range(i + 1, len(values) - 1):
                    a = values[i - 1]
                    b = values[i]
                    c = values[j]
                    d = values[j + 1]

                    if self.__is_cost_lower_after_swap(a, b, c, d):
                        values[b:d] = reversed(values[b:d])
                        improved = True

        return values

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
        improved = True

        while improved:
            improved = False

            for i in range(len(values) - 1):
                for j in range(i + 1, len(values)):
                    swapped_values = self.__swap(values, i, j)

                    if self.__compare(values, swapped_values):
                        values = swapped_values
                        improved = True

        return values

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
        [a, b] = np.random.choice(list(range(len(values))))
        values[a], values[b] = values[b], values[a]

        return values
