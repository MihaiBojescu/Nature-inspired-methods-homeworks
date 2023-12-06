import random
import typing as t
import numpy as np
from algorithms.binary_genetic_algorithm.individual import DecodedIndividual
from util.sort import quicksort
from algorithms.binary_genetic_algorithm.population import correct_population


def tournament_selection(
    fitness_compare_function: t.Callable[[any, any], bool],
    tournament_size: int,
) -> t.Callable[[t.List[DecodedIndividual]], t.List[DecodedIndividual]]:
    def run(population: t.List[DecodedIndividual]):
        population = correct_population(population=population)

        selected_parents = []

        while len(selected_parents) < len(population):
            tournament = random.sample(population=population, k=tournament_size)
            tournament_sorted = quicksort(
                data=tournament,
                comparator=lambda a, b: fitness_compare_function(a[1], b[1]),
            )
            selected_parents.append(tournament_sorted[0])

        return selected_parents

    return run


def roulette_wheel_selection(
    target: t.Union[t.Literal["maximise"], t.Literal["minimise"]]
):
    def run(
        population: t.List[DecodedIndividual],
    ) -> t.List[DecodedIndividual]:
        population = correct_population(population)

        fitness_sum = np.sum(individual[1] for individual in population)
        selection_probabilities = (
            [individual[1] / fitness_sum for individual in population]
            if target == "maximise"
            else [1 / individual[1] / fitness_sum for individual in population]
        )
        selected_individuals = []
        selected_individual = None

        for _ in range(0, len(population)):
            selected_probability = np.random.uniform(low=0, high=1)

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

    return run
