import random
import typing as t
from algorithms.apso.operators import BaseTwoOptOperator, BasePathLinkerOperator, BaseSwapOperator
from algorithms.apso.individual import Individual
from functions.combinatorial.tsp.util.common import TspResult

MTSPSolution = t.List[t.List[int]]


class TwoOptOperator(BaseTwoOptOperator[MTSPSolution, TspResult]):
    __fitness_function: t.Callable[[MTSPSolution], TspResult]
    __fitness_compare_function: t.Callable[[TspResult, TspResult], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[MTSPSolution], TspResult],
        fitness_compare_function: t.Callable[[TspResult, TspResult], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, individual: Individual[MTSPSolution, TspResult]):
        for segment in individual.position:
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

        return individual

    def __is_cost_lower_edge_swap(self, a: int, b: int, c: int, d: int):
        original_cost = self.__fitness_function([[a, b]]) + self.__fitness_function([[c, d]])
        swapped_cost = self.__fitness_function([[a, c]]) + self.__fitness_function([[b, d]])

        return self.__fitness_compare_function(swapped_cost, original_cost)

    def __is_cost_lower_instance_swap(self, values: t.List[int], swapped_values: t.List[int]):
        original_cost = self.__fitness_function([values])
        swapped_cost = self.__fitness_function([swapped_values])

        return self.__fitness_compare_function(swapped_cost, original_cost)


class PathLinkerOperator(BasePathLinkerOperator[MTSPSolution, TspResult]):
    __fitness_function: t.Callable[[MTSPSolution], TspResult]
    __fitness_compare_function: t.Callable[[TspResult, TspResult], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[MTSPSolution], TspResult],
        fitness_compare_function: t.Callable[[TspResult, TspResult], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, individual: Individual[MTSPSolution, TspResult]):
        for segment_index, _ in enumerate(individual.position):
            improved = True
            values_without_segment = [
                current_segment
                for current_segment in individual.position
                if current_segment == individual.position[segment_index]
            ]

            while improved:
                improved = False
                segment = individual.position[segment_index]

                for i in range(len(segment) - 1):
                    for j in range(i + 1, len(segment)):
                        swapped_values = self.__swap(segment, i, j)

                        if not self.__is_cost_lower_instance_swap([segment], [swapped_values]):
                            continue

                        if not self.__is_cost_lower_instance_swap(
                            [*values_without_segment, segment], [*values_without_segment, swapped_values]
                        ):
                            continue

                        individual.position[segment_index] = swapped_values
                        improved = True

        return individual

    def __swap(self, values: t.List[int], i: int, j: int):
        swapped_values = values.copy()
        swapped_values[i], swapped_values[j] = swapped_values[j], swapped_values[i]

        return swapped_values

    def __is_cost_lower_instance_swap(self, original_values: MTSPSolution, swapped_values: MTSPSolution):
        original_cost = self.__fitness_function(original_values)
        swapped_cost = self.__fitness_function(swapped_values)

        return self.__fitness_compare_function(swapped_cost, original_cost)


class SwapOperator(BaseSwapOperator[MTSPSolution, TspResult]):
    def run(self, individual: Individual[MTSPSolution, TspResult]):
        [segment_1, segment_2] = random.sample(range(len(individual.position)), k=2)
        value_1 = random.randint(a=0, b=len(individual.position[segment_1]) - 1)
        value_2 = random.randint(a=0, b=len(individual.position[segment_2]) - 1)

        individual.position[segment_1][value_1], individual.position[segment_2][value_2] = (
            individual.position[segment_2][value_2],
            individual.position[segment_1][value_1],
        )

        return individual
