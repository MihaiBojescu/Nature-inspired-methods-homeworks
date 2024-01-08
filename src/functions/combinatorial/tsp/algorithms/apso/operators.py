import random
import typing as t
from algorithms.apso.operators import BaseTwoOptOperator, BasePathRelinkingOperator, BaseSwapOperator
from algorithms.apso.individual import Individual
from functions.combinatorial.tsp.util.common import MTSPResult

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
        i, j = swap_indices
        values = [value.copy() for value in individual.position]
        values[i][j] = best_individual.position[i][j]
        cost = self.__fitness_function(values)

        return values, cost


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
