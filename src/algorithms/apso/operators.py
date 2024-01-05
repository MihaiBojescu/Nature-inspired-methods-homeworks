import typing as t
import numpy as np

T = t.TypeVar("T")


class BaseOperator:
    def run(self, values: t.List[t.List[T]]) -> t.List[np.int64]:
        return []


class TwoOptOperator(BaseOperator):
    __fitness_function: t.Callable[[t.List[np.int64]], np.float32]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[t.List[np.int64]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, values: t.List[t.List[np.uint64]]):
        values_copy = [segment.copy() for segment in values]

        for segment in values_copy:
            improved = True

            while improved:
                improved = False

                for i in range(1, len(segment) - 2):
                    for j in range(i + 1, len(segment) - 1):
                        a = segment[i - 1]
                        b = segment[i]
                        c = segment[j]
                        d = segment[j + 1]
                        b_pos = i
                        d_pos = j + 1

                        if not self.__is_cost_lower_edge_swap(a, b, c, d):
                            continue

                        swapped_segment = segment.copy()
                        swapped_segment[b_pos:d_pos] = np.flip(segment[b_pos:d_pos])

                        if not self.__is_cost_lower_instance_swap(segment, swapped_segment):
                            continue

                        segment[b_pos:d_pos] = np.flip(segment[b_pos:d_pos])
                        improved = True

        return values_copy

    def __is_cost_lower_edge_swap(self, a: np.int64, b: np.int64, c: np.int64, d: np.int64):
        original_cost = self.__fitness_function([[a, b]]) + self.__fitness_function([[c, d]])
        swapped_cost = self.__fitness_function([[a, c]]) + self.__fitness_function([[b, d]])

        return self.__fitness_compare_function(swapped_cost, original_cost)

    def __is_cost_lower_instance_swap(self, values: t.List[np.int64], swapped_values: t.List[np.int64]):
        original_cost = self.__fitness_function([values])
        swapped_cost = self.__fitness_function([swapped_values])

        return self.__fitness_compare_function(swapped_cost, original_cost)


class PathLinkerOperator(BaseOperator):
    __fitness_function: t.Callable[[t.List[np.int64]], np.float32]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[t.List[np.int64]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, values: t.List[t.List[np.int64]]):
        values_copy = [segment.copy() for segment in values]

        for segment_index, _ in enumerate(values_copy):
            values_without_segment = [
                current_segment
                for current_segment in values_copy
                if np.array_equal(current_segment, values_copy[segment_index])
            ]
            improved = True

            while improved:
                improved = False
                segment = values_copy[segment_index]

                for i in range(len(segment) - 1):
                    for j in range(i + 1, len(segment)):
                        swapped_values = self.__swap(segment, i, j)

                        if not self.__is_cost_lower_instance_swap([segment], [swapped_values]):
                            continue

                        if not self.__is_cost_lower_instance_swap(
                            [*values_without_segment, segment], [*values_without_segment, swapped_values]
                        ):
                            continue

                        values_copy[segment_index] = swapped_values
                        improved = True

        return values_copy

    def __swap(self, values: t.List[np.int64], i: int, j: int):
        swapped_values = values.copy()
        swapped_values[i], swapped_values[j] = swapped_values[j], swapped_values[i]

        return swapped_values

    def __is_cost_lower_instance_swap(
        self, original_values: t.List[t.List[np.int64]], swapped_values: t.List[t.List[np.int64]]
    ):
        original_cost = self.__fitness_function(original_values)
        swapped_cost = self.__fitness_function(swapped_values)

        return self.__fitness_compare_function(swapped_cost, original_cost)


class SwapOperator(BaseOperator):
    def run(self, values: t.List[t.List[np.int64]]):
        values_copy = values.copy()

        [segment_1, segment_2] = np.random.choice(list(range(len(values_copy))), size=2, replace=False)
        value_1 = np.random.randint(low=0, high=len(values_copy[segment_1]))
        value_2 = np.random.randint(low=0, high=len(values_copy[segment_2]))

        values_copy[segment_1][value_1], values_copy[segment_2][value_2] = (
            values_copy[segment_2][value_2],
            values_copy[segment_1][value_1],
        )

        return values_copy
