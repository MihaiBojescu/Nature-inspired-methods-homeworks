import random
import typing as t
from algorithms.ga.operators import BaseCrossoverOperator, BaseMutationOperator, BaseSelectionOperator
from algorithms.ga.individual import Individual
from functions.combinatorial.tsp.algorithms.ga.encoder import Encoder
from util.sort import quicksort

MTSPSolution = t.List[t.List[int]]


class CrossoverOperator(BaseCrossoverOperator[MTSPSolution]):
    __encoder: Encoder

    def __init__(self, encoder: Encoder):
        self.__encoder = encoder

    def run(
        self, parent_1: Individual[MTSPSolution], parent_2: Individual[MTSPSolution]
    ) -> t.Tuple[Individual[MTSPSolution], Individual[MTSPSolution]]:
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


class SwapMutationOperator(BaseMutationOperator[MTSPSolution]):
    def run(self, child: Individual[MTSPSolution]) -> Individual[MTSPSolution]:
        [segment_1, segment_2] = random.sample(range(len(child)), k=2)
        value_1 = random.randint(a=0, b=len(child[segment_1]))
        value_2 = random.randint(a=0, b=len(child[segment_2]))

        child[segment_1][value_1], child[segment_2][value_2] = (
            child[segment_2][value_2],
            child[segment_1][value_1],
        )

        return child


class TournamentSelectionOperator(BaseSelectionOperator[MTSPSolution]):
    __fitness_compare_function: t.Callable[[float, float], bool]
    __tournament_size: int

    def __init__(
        self,
        fitness_compare_function: t.Callable[[float, float], bool],
        tournament_size: int,
    ):
        self.__fitness_compare_function = fitness_compare_function
        self.__tournament_size = tournament_size

    def run(self, population: t.List[Individual[MTSPSolution]]) -> t.List[Individual[MTSPSolution]]:
        selected_parents = []

        while len(selected_parents) < len(population):
            tournament = random.sample(population=population, k=self.__tournament_size)
            tournament_sorted = quicksort(
                data=tournament,
                comparator=lambda a, b: self.__fitness_compare_function(a.fitness, b.fitness),
            )
            selected_parents.append(tournament_sorted[0])

        return selected_parents


class RouletteWheelSelectionOperator(BaseMutationOperator[MTSPSolution]):
    __target: t.Union[t.Literal["maximise"], t.Literal["minimise"]]

    def __init__(
        self,
        target: t.Union[t.Literal["maximise"], t.Literal["minimise"]],
    ):
        self.__target = target

    def run(
        self,
        population: t.List[Individual[MTSPSolution]],
    ) -> MTSPSolution:
        fitness_sum = sum([individual.fitness for individual in population])
        selection_probabilities = (
            [individual.fitness / fitness_sum for individual in population]
            if self.__target == "maximise"
            else [1 / fitness / fitness_sum for fitness in population]
        )
        selected_individuals: t.List[Individual[MTSPSolution]] = []
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
