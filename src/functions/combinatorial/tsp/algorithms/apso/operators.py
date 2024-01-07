import random
import typing as t
from algorithms.apso.operators import BaseTwoOptOperator, BasePathLinkerOperator, BaseSwapOperator


class TwoOptOperator(BaseTwoOptOperator[t.List[t.List[int]]]):
    __fitness_function: t.Callable[[t.List[t.List[int]]], float]
    __fitness_compare_function: t.Callable[[float, float], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[t.List[t.List[int]]], float],
        fitness_compare_function: t.Callable[[float, float], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, values: t.List[t.List[int]]):
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
                        swapped_segment[b_pos:d_pos] = reversed(segment[b_pos:d_pos])

                        if not self.__is_cost_lower_instance_swap(segment, swapped_segment):
                            continue

                        segment[b_pos:d_pos] = reversed(segment[b_pos:d_pos])
                        improved = True

        return values_copy

    def __is_cost_lower_edge_swap(self, a: int, b: int, c: int, d: int):
        original_cost = self.__fitness_function([[a, b]]) + self.__fitness_function([[c, d]])
        swapped_cost = self.__fitness_function([[a, c]]) + self.__fitness_function([[b, d]])

        return self.__fitness_compare_function(swapped_cost, original_cost)

    def __is_cost_lower_instance_swap(self, values: t.List[int], swapped_values: t.List[int]):
        original_cost = self.__fitness_function([values])
        swapped_cost = self.__fitness_function([swapped_values])

        return self.__fitness_compare_function(swapped_cost, original_cost)


class PathLinkerOperator(BasePathLinkerOperator[t.List[t.List[int]]]):
    __fitness_function: t.Callable[[t.List[t.List[int]]], float]
    __fitness_compare_function: t.Callable[[float, float], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[t.List[t.List[int]]], float],
        fitness_compare_function: t.Callable[[float, float], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, values: t.List[t.List[int]]):
        values_copy = [segment.copy() for segment in values]

        for segment_index, _ in enumerate(values_copy):
            values_without_segment = [
                current_segment for current_segment in values_copy if current_segment == values_copy[segment_index]
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

    def __swap(self, values: t.List[int], i: int, j: int):
        swapped_values = values.copy()
        swapped_values[i], swapped_values[j] = swapped_values[j], swapped_values[i]

        return swapped_values

    def __is_cost_lower_instance_swap(self, original_values: t.List[t.List[int]], swapped_values: t.List[t.List[int]]):
        original_cost = self.__fitness_function(original_values)
        swapped_cost = self.__fitness_function(swapped_values)

        return self.__fitness_compare_function(swapped_cost, original_cost)


class SwapOperator(BaseSwapOperator[t.List[t.List[int]]]):
    def run(self, values: t.List[t.List[int]]):
        values_copy = values.copy()

        [segment_1, segment_2] = random.sample(range(len(values_copy)), k=2)
        value_1 = random.randint(a=0, b=len(values_copy[segment_1]) - 1)
        value_2 = random.randint(a=0, b=len(values_copy[segment_2]) - 1)

        values_copy[segment_1][value_1], values_copy[segment_2][value_2] = (
            values_copy[segment_2][value_2],
            values_copy[segment_1][value_1],
        )

        return values_copy
