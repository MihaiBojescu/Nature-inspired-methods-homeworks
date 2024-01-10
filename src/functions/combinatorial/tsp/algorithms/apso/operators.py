import random
import typing as t
from algorithms.apso.operators import BaseTwoOptOperator, BasePathRelinkingOperator, BaseSwapOperator
from algorithms.apso.individual import Individual
from functions.combinatorial.tsp.util.common import MTSPResult
from functions.combinatorial.tsp.util.encoder import Encoder

MTSPSolution = t.List[t.List[int]]


class TwoOptOperator(BaseTwoOptOperator[MTSPSolution, MTSPResult]):
    __fitness_function: t.Callable[[MTSPSolution], MTSPResult]
    __fitness_compare_function: t.Callable[[MTSPResult, MTSPResult], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[MTSPSolution], MTSPResult],
        fitness_compare_function: t.Callable[[MTSPResult, MTSPResult], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(self, individual: Individual[MTSPSolution, MTSPResult]):
        values = [segment.copy() for segment in individual.position]

        for segment in values:
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

        individual.position = values
        return individual

    def __is_cost_lower_edge_swap(self, a: int, b: int, c: int, d: int):
        original_cost = self.__fitness_function([[a, b]]) + self.__fitness_function([[c, d]])
        swapped_cost = self.__fitness_function([[a, c]]) + self.__fitness_function([[b, d]])

        return self.__fitness_compare_function(swapped_cost, original_cost)

    def __is_cost_lower_instance_swap(self, values: t.List[int], swapped_values: t.List[int]):
        original_cost = self.__fitness_function([values])
        swapped_cost = self.__fitness_function([swapped_values])

        return self.__fitness_compare_function(swapped_cost, original_cost)


class PathRelinkingOperator(BasePathRelinkingOperator[MTSPSolution, MTSPResult]):
    __fitness_function: t.Callable[[MTSPSolution], MTSPResult]
    __fitness_compare_function: t.Callable[[MTSPResult, MTSPResult], bool]

    def __init__(
        self,
        fitness_function: t.Callable[[MTSPSolution], MTSPResult],
        fitness_compare_function: t.Callable[[MTSPResult, MTSPResult], bool],
    ):
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function

    def run(
        self, individual: Individual[MTSPSolution, MTSPResult], best_individual: Individual[MTSPSolution, MTSPResult]
    ):
        symmetric_difference_indices = [
            (i, j)
            for i in range(len(individual.position))
            for j in range(len(individual.position[i]))
            if individual.position[i][j] != best_individual.position[i][j]
        ]
        best_values, best_cost = (
            (individual.position, individual.fitness)
            if self.__fitness_compare_function(individual.fitness, best_individual.fitness)
            else (best_individual.position, best_individual.fitness)
        )

        while len(symmetric_difference_indices) > 0:
            current_moves = [
                self.__build_swapped_solution(
                    individual=individual, best_individual=best_individual, swap_indices=indices
                )
                for indices in symmetric_difference_indices
            ]
            current_best_move_index = current_moves.index(min(current_moves, key=lambda entry: entry[1]))
            symmetric_difference_indices.pop(current_best_move_index)
            current_best_values, current_best_values_cost = current_moves[current_best_move_index]

            if self.__fitness_compare_function(current_best_values_cost, best_cost):
                best_values = current_best_values
                best_cost = current_best_values_cost

        individual.position = best_values
        return individual

    def __build_swapped_solution(
        self,
        individual: Individual[MTSPSolution, MTSPResult],
        best_individual: Individual[MTSPSolution, MTSPResult],
        swap_indices: t.Tuple[int, int],
    ) -> t.Tuple[MTSPSolution, MTSPResult]:
        values = [value.copy() for value in individual.position]
        i_new_value, j_new_value = swap_indices
        i_old_value, j_old_value = self.__find_indices(
            haystack=individual, needle=best_individual.position[i_new_value][j_new_value]
        )
        old_value = values[i_new_value][j_new_value]

        values[i_new_value][j_new_value], values[i_old_value][j_old_value] = (
            best_individual.position[i_new_value][j_new_value],
            old_value,
        )
        cost = self.__fitness_function(values)

        return values, cost

    def __find_indices(self, haystack: Individual[MTSPSolution, MTSPResult], needle: int) -> t.Tuple[int, int]:
        for i, row in enumerate(haystack.position):
            for j, value in enumerate(row):
                if value == needle:
                    return (i, j)

        return -1, -1


class SwapOperator(BaseSwapOperator[MTSPSolution, MTSPResult]):
    def run(self, individual: Individual[MTSPSolution, MTSPResult]):
        [segment_1, segment_2] = random.sample(range(len(individual.position)), k=2)
        value_1 = random.randint(a=0, b=len(individual.position[segment_1]) - 1)
        value_2 = random.randint(a=0, b=len(individual.position[segment_2]) - 1)

        individual.position[segment_1][value_1], individual.position[segment_2][value_2] = (
            individual.position[segment_2][value_2],
            individual.position[segment_1][value_1],
        )

        return individual

class ComplexSwapOperator(BaseSwapOperator[MTSPSolution, MTSPResult]):
    __encoder: Encoder
    __probability: float

    def __init__(self, encoder: Encoder, probability: float):
        self.__encoder = encoder
        self.__probability = probability

    def run(self, individual: Individual[MTSPSolution, MTSPResult]) -> Individual[MTSPSolution, MTSPResult]:
        values = self.__encoder.encode(value=individual.position)

        values = self.__swap(values)
        values = self.__reverse_swap(values)
        values = self.__slide(values)

        individual.position = self.__encoder.decode(value=values)
        return individual

    def __swap(self, values: t.List[int]) -> t.List[int]:
        if random.random() > self.__probability:
            return values

        a = random.randint(a=0, b=len(values) - 1)
        b = random.randint(a=0, b=len(values) - 1)

        values[a], values[b] = values[b], values[a]

        return values

    def __reverse_swap(self, values: t.List[int]) -> t.List[int]:
        if random.random() > self.__probability:
            return values

        a = random.randint(a=0, b=len(values) - 1)
        b = random.randint(a=0, b=len(values) - 1)

        values[a:b] = reversed(values[a:b])

        return values

    def __slide(self, values: t.List[int]) -> t.List[int]:
        if random.random() > self.__probability:
            return values

        a = random.randint(a=0, b=len(values) - 1)
        b = random.randint(a=0, b=len(values) - 1)

        value = values.pop(a)
        values.insert(b, value)

        return values
