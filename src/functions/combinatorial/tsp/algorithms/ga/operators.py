import random
import typing as t
from algorithms.ga.operators import BaseCrossoverOperator, BaseMutationOperator, BaseSelectionOperator
from algorithms.ga.individual import Individual
from functions.combinatorial.tsp.algorithms.ga.encoder import Encoder
from functions.combinatorial.tsp.util.common import TspResult
from util.sort import quicksort

MTSPSolution = t.List[t.List[int]]


class TournamentSelectionOperator(BaseSelectionOperator[MTSPSolution, TspResult]):
    __fitness_compare_function: t.Callable[[float, float], bool]
    __tournament_size: int

    def __init__(
        self,
        fitness_compare_function: t.Callable[[float, float], bool],
        tournament_size: int,
    ):
        self.__fitness_compare_function = fitness_compare_function
        self.__tournament_size = tournament_size

    def run(
        self, population: t.List[Individual[MTSPSolution, TspResult]]
    ) -> t.List[Individual[MTSPSolution, TspResult]]:
        selected_parents = []

        while len(selected_parents) < len(population):
            tournament = random.sample(population=population, k=self.__tournament_size)
            tournament_sorted = quicksort(
                data=tournament,
                comparator=lambda a, b: self.__fitness_compare_function(a.fitness, b.fitness),
            )
            selected_parents.append(tournament_sorted[0])

        return selected_parents


class RouletteWheelSelectionOperator(BaseMutationOperator[MTSPSolution, TspResult]):
    __target: t.Union[t.Literal["maximise"], t.Literal["minimise"]]

    def __init__(
        self,
        target: t.Union[t.Literal["maximise"], t.Literal["minimise"]],
    ):
        self.__target = target

    def run(
        self,
        population: t.List[Individual[MTSPSolution, TspResult]],
    ) -> MTSPSolution:
        fitness_sum = sum(individual.fitness.optimal_cost for individual in population)
        selection_probabilities = (
            [individual.fitness.optimal_cost / fitness_sum for individual in population]
            if self.__target == "maximise"
            else [1 / individual.fitness.optimal_cost / fitness_sum for individual in population]
        )
        selected_individuals: t.List[Individual[MTSPSolution, TspResult]] = []
        selected_individual = None

        for _ in range(0, len(population)):
            selected_probability = random.uniform(a=0, b=1)

            for index in range(0, len(selection_probabilities) - 1):
                current_probability = selection_probabilities[index]
                next_probability = selection_probabilities[index + 1]

                if current_probability <= selected_probability <= next_probability:
                    selected_individual = population[index]
                    break

            if selected_individual is None:
                selected_individual = population[-1]

            selected_individuals.append(selected_individual)

        return selected_individuals


class CrossoverOperator(BaseCrossoverOperator[MTSPSolution, TspResult]):
    __encoder: Encoder
    __fitness_function: t.Callable[[MTSPSolution], TspResult]

    def __init__(self, encoder: Encoder, fitness_function: t.Callable[[MTSPSolution], TspResult]):
        self.__encoder = encoder
        self.__fitness_function = fitness_function

    def run(
        self, parent_1: Individual[MTSPSolution, TspResult], parent_2: Individual[MTSPSolution, TspResult]
    ) -> t.Tuple[Individual[MTSPSolution, TspResult], Individual[MTSPSolution, TspResult]]:
        """
        Implements Crossover from https://www.growingscience.com/dsl/Vol10/dsl_2021_22.pdf. Order of operations is
        reversed due to encoding.
        """
        n = sum(len(tour) for tour in parent_1.genes)

        parent_1_encoded = self.__encoder.encode(parent_1.genes)
        parent_2_encoded = self.__encoder.encode(parent_2.genes)
        child_1 = [0 for _ in range(n)]
        child_2 = [0 for _ in range(n)]

        child_1[0] = parent_2_encoded[-1]
        child_2[0] = parent_1_encoded[-1]
        child_1[-1] = parent_2_encoded[0]
        child_2[-1] = parent_1_encoded[0]

        for i in range(0, n):
            for j in range(1, n - 1):
                if parent_2_encoded[i] == parent_1_encoded[j]:
                    child_1[j] = parent_2_encoded[j]

                if parent_1_encoded[i] == parent_2_encoded[j]:
                    child_2[j] = parent_1_encoded[j]

        child_1 = self.__encoder.decode(child_1)
        child_2 = self.__encoder.decode(child_2)

        return Individual(genes=child_1, fitness_function=self.__fitness_function), Individual(
            genes=child_2, fitness_function=self.__fitness_function
        )


class SwapMutationOperator(BaseMutationOperator[MTSPSolution, TspResult]):
    __probability: float

    def __init__(self, probability: float):
        self.__probability = probability

    def run(self, child: Individual[MTSPSolution, TspResult]) -> Individual[MTSPSolution, TspResult]:
        values = [segment.copy() for segment in child.genes]
        runs = sum(len(segment) for segment in values)

        for _ in range(runs):
            if random.random() > self.__probability:
                continue

            segment_1 = random.randint(
                a=0,
                b=len(values) - 1,
            )
            segment_2 = random.randint(
                a=0,
                b=len(values) - 1,
            )
            value_1 = random.randint(a=0, b=len(values[segment_1]) - 1)
            value_2 = random.randint(a=0, b=len(values[segment_2]) - 1)

            values[segment_1][value_1], values[segment_2][value_2] = (
                values[segment_2][value_2],
                values[segment_1][value_1],
            )

        child.genes = values
        return child
