from dataclasses import dataclass, field
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
    optimal_min_max: TspResultOptimalMinMax = field(
        default_factory=TspResultOptimalMinMax
    )
    optimal_cost: float = 0


def segment_instances(
    fn: t.Callable[[t.List[t.List[City]]], TspResult], dimensions: int
) -> t.Callable[[t.List[City]], TspResult]:
    if dimensions < 1:
        raise IndexError("Dimension < 1 in segment function")

    sections = None

    def run(salesmen_routes: t.List[t.List[City]]) -> TspResult:
        nonlocal sections

        if sections is None:
            sections = [0]
            current_index = 0
            max_size = (len(salesmen_routes) // 2) - 1

            for _ in range(dimensions - 1):
                size = np.random.randint(
                    low=1,
                    high=max_size,
                )
                sections.append(current_index + size)
                current_index += size

            sections.append(len(salesmen_routes))

        salesmen_routes = [
            salesmen_routes[sections[i] : sections[i + 1]] for i in range(len(sections) - 1)
        ]
        result = fn(salesmen_routes)
        return result

    return run


def augment_instances(
    fn: t.Callable[[t.List[t.List[City]]], TspResult], home_city: City
) -> t.Callable[[t.List[t.List[City]]], TspResult]:
    def run(salesmen_routes: t.List[t.List[City]]) -> TspResult:
        for i, _ in enumerate(salesmen_routes):
            salesmen_routes[i] = np.concatenate(([home_city], salesmen_routes[i]))

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
        total_cost += __calculate_distance(salesman_route[i], salesman_route[i + 1])

    return total_cost


def __calculate_distance(a: City, b: City) -> float:
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class InitialPopulationGenerator:
    __function_definition: CombinatorialFunctionDefinition
    __population_size: int

    def __init__(
        self,
        function_definition: CombinatorialFunctionDefinition,
        population_size: int,
    ):
        self.__function_definition = function_definition
        self.__population_size = population_size

    def __call__(self):
        return [
            np.random.choice(
                [city.identifier for city in self.__function_definition.values[1:]],
                size=len(self.__function_definition.values) - 1,
                replace=False,
            )
            for _ in range(self.__population_size)
        ]
