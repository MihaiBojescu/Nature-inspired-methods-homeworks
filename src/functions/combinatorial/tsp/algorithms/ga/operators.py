import random
import typing as t
from algorithms.ga.operators import BaseCrossoverOperator, BaseMutationOperator, BaseSelectionOperator
from algorithms.ga.individual import Individual
from functions.combinatorial.tsp.algorithms.ga.encoder import Encoder
from functions.combinatorial.tsp.util.common import MTSPResult
from util.sort import quicksort

MTSPSolution = t.List[t.List[int]]


class TournamentSelectionOperator(BaseSelectionOperator[MTSPSolution, MTSPResult]):
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
        self, population: t.List[Individual[MTSPSolution, MTSPResult]]
    ) -> t.List[Individual[MTSPSolution, MTSPResult]]:
        selected_parents = []

        while len(selected_parents) < len(population):
            tournament = random.sample(population=population, k=self.__tournament_size)
            tournament_sorted = quicksort(
                data=tournament,
                comparator=lambda a, b: self.__fitness_compare_function(a.fitness, b.fitness),
            )
            selected_parents.append(tournament_sorted[0])

        return selected_parents


class RouletteWheelSelectionOperator(BaseMutationOperator[MTSPSolution, MTSPResult]):
    __target: t.Union[t.Literal["maximise"], t.Literal["minimise"]]

    def __init__(
        self,
        target: t.Union[t.Literal["maximise"], t.Literal["minimise"]],
    ):
        self.__target = target

    def run(
        self,
        population: t.List[Individual[MTSPSolution, MTSPResult]],
    ) -> MTSPSolution:
        fitness_sum = sum(individual.fitness.optimal_cost for individual in population)
        selection_probabilities = (
            [individual.fitness.optimal_cost / fitness_sum for individual in population]
            if self.__target == "maximise"
            else [(fitness_sum - individual.fitness.optimal_cost) / fitness_sum for individual in population]
        )
        selection_probabilities_max = max(selection_probabilities)
        selection_probabilities_min = min(selection_probabilities)
        selection_probabilities = (
            [1 for _ in range(len(selection_probabilities))]
            if selection_probabilities_max == selection_probabilities_min
            else [
                (probability - selection_probabilities_min)
                / (selection_probabilities_max - selection_probabilities_min)
                for probability in selection_probabilities
            ]
        )

        maximise_probability_comparator = (
            lambda current, selected, next: current_probability <= selected_probability <= next_probability
        )
        minimise_probability_comparator = (
            lambda current, selected, next: current_probability >= selected_probability >= next_probability
        )
        probability_comparator = (
            maximise_probability_comparator if self.__target == "maximise" else minimise_probability_comparator
        )

        selected_individuals: t.List[Individual[MTSPSolution, MTSPResult]] = []
        selected_individual = None

        for _ in range(0, len(population)):
            selected_probability = random.uniform(a=0, b=1)

            for index in range(0, len(selection_probabilities) - 1):
                current_probability = selection_probabilities[index]
                next_probability = selection_probabilities[index + 1]

                if probability_comparator(current_probability, selected_probability, next_probability):
                    selected_individual = population[index]
                    break

            if selected_individual is None:
                selected_individual = population[-1]

            selected_individuals.append(selected_individual)

        return selected_individuals


class CrossoverOperator(BaseCrossoverOperator[MTSPSolution, MTSPResult]):
    __encoder: Encoder
    __fitness_function: t.Callable[[MTSPSolution], MTSPResult]

    def __init__(self, encoder: Encoder, fitness_function: t.Callable[[MTSPSolution], MTSPResult]):
        self.__encoder = encoder
        self.__fitness_function = fitness_function

    def run(
        self, parent_1: Individual[MTSPSolution, MTSPResult], parent_2: Individual[MTSPSolution, MTSPResult]
    ) -> t.Tuple[Individual[MTSPSolution, MTSPResult], Individual[MTSPSolution, MTSPResult]]:
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


class ComplexMutationOperator(BaseMutationOperator[MTSPSolution, MTSPResult]):
    __encoder: Encoder
    __probability: float

    def __init__(self, encoder: Encoder, probability: float):
        self.__encoder = encoder
        self.__probability = probability

    def run(self, child: Individual[MTSPSolution, MTSPResult]) -> Individual[MTSPSolution, MTSPResult]:
        if random.random() > self.__probability:
            return child

        values = self.__encoder.encode(value=child.genes)

        values = self.__swap(values)
        values = self.__reverse_swap(values)
        values = self.__slide(values)

        child.genes = self.__encoder.decode(value=values)
        return child

    def __swap(self, values: t.List[int]) -> t.List[int]:
        a = random.randint(a=0, b=len(values) - 1)
        b = random.randint(a=0, b=len(values) - 1)

        values[a], values[b] = values[b], values[a]

        return values

    def __reverse_swap(self, values: t.List[int]) -> t.List[int]:
        a = random.randint(a=0, b=len(values) - 1)
        b = random.randint(a=0, b=len(values) - 1)

        values[a:b] = reversed(values[a:b])

        return values

    def __slide(self, values: t.List[int]) -> t.List[int]:
        a = random.randint(a=0, b=len(values) - 1)
        b = random.randint(a=0, b=len(values) - 1)

        value = values.pop(a)
        values.insert(b, value)

        return values
