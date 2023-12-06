import math
import typing as t
import numpy as np
from algorithms.binary_genetic_algorithm.individual import DecodedIndividual


def correct_population(population: t.List[DecodedIndividual]):
    filtered_population = list(
        filter(lambda individual: is_not_undefined_value(individual[1]), population)
    )

    if len(filtered_population) == 0:
        return []

    for _ in range(len(filtered_population), len(population)):
        selected_individual_index = np.random.randint(
            low=0, high=len(filtered_population)
        )
        selected_individual = filtered_population[selected_individual_index]
        filtered_population.append(selected_individual)

    return filtered_population


def is_not_undefined_value(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)
