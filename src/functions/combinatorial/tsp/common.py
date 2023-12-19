from dataclasses import dataclass
import numpy as np
import typing as t
from functions.combinatorial.definition import CombinatorialFunctionDefinition


@dataclass
class City:
    identifier: int
    x: float
    y: float


@dataclass
class TspResultOptimalMinMax:
    min: float = +np.inf
    max: float = -np.inf


@dataclass
class TspResult:
    optimal_min_max: TspResultOptimalMinMax = TspResultOptimalMinMax()
    optimal_cost: float = 0


def augment_instances(
    fn: t.Callable[[t.List[t.List[City]]], TspResult], home_city: City
) -> t.Callable[[t.List[t.List[City]]], TspResult]:
    def run(salesmen_routes: t.List[t.List[City]]) -> TspResult:
        for i in range(salesmen_routes):
            salesmen_routes[i].insert(0, home_city)

        result = fn(salesmen_routes)
        return result

    return run


def min_max_multiple_tsp(salesmen_routes: t.List[t.List[City]]) -> TspResult:
    result = TspResult()

    for salesman_route in salesmen_routes:
        cost = tsp(salesman_route)

        result.optimal_min_max.min = min(result.optimal_min_max.min, cost)
        result.optimal_min_max.max = max(result.optimal_min_max.max, cost)
        result.optimal_cost += cost

    return result


def tsp(salesman_route: t.List[City]) -> float:
    total_cost = 0

    for i in range(len(salesman_route) - 1):
        total_cost += __calculate_distance(salesman_route[i], salesman_route[i] + 1)

    return total_cost


def __calculate_distance(a: City, b: City) -> float:
    return np.sqrt(((a.x - b.x) ^ 2 + (a.y - b.y) ^ 2))


def generate_initial_population(
    function_definition: CombinatorialFunctionDefinition,
    population_size: int,
    dimensions: int,
):
    def run():
        population = []

        pickable_cities = len(function_definition.values) - 1
        tour_max_pickable_cities = (pickable_cities // 2) - 1
        unpicked_cities = np.random.choice(
            function_definition.values[1:], size=pickable_cities, replace=False
        )

        for _ in range(population_size):
            unpicked_cities_copy = unpicked_cities.copy()
            picked_city_tours = []

            for _ in range(dimensions - 1):
                picked_tour_size = np.random.randint(
                    low=1,
                    high=min(
                        len(unpicked_cities_copy),
                        tour_max_pickable_cities,
                    ),
                )
                picked_cities, unpicked_cities_copy = (
                    unpicked_cities_copy[:picked_tour_size],
                    unpicked_cities_copy[picked_tour_size:],
                )
                picked_city_tours.append(picked_cities)

            picked_city_tours.append(unpicked_cities_copy)
            population.append(picked_city_tours)

        return population

    return run
