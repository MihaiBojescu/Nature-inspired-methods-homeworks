from dataclasses import dataclass, field
import math
import numpy as np
import numpy.typing as npt
import typing as t
from functions.combinatorial.definition import CombinatorialFunctionDefinition


@dataclass
class City:
    identifier: int
    x: float
    y: float


class CostCalculator:
    def __call__(self, cities: t.List[City]):
        result = [[0.0 for _ in range(len(cities))] for _ in range(len(cities))]

        for i in range(0, len(cities) - 1):
            for j in range(i + 1, len(cities)):
                cost = self.__calculate_distance(cities[i], cities[j])
                result[i][j] = cost
                result[j][i] = cost

        return result

    def __calculate_distance(self, a: City, b: City) -> float:
        return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


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

    def __gt__(self, other):
        return self.optimal_cost > other.optimal_cost

    def __eq__(self, other):
        return self.optimal_cost == other.optimal_cost

    def __lt__(self, other):
        return self.optimal_cost < other.optimal_cost


class InstanceSegmenter:
    __fn: t.Callable[[t.List[t.List[City]]], TspResult]
    __salesmen: int
    __tours: t.Optional[t.List[int]]

    def __init__(
        self,
        fn: t.Callable[[t.List[t.List[City]]], TspResult],
        salesmen: int,
    ):
        if salesmen < 1:
            raise IndexError("Salesmen < 1 in segment function")

        self.__fn = fn
        self.__salesmen = salesmen
        self.__tours = None

    def __call__(self, salesmen_routes: npt.NDArray[np.int64]) -> TspResult:
        if self.__tours is None:
            tours = []
            unpicked_cities = salesmen_routes.copy()

            for salesman in range(self.__salesmen, 1, -1):
                max_tour_size = math.floor(
                    ((len(unpicked_cities) - 1 + salesman) / salesman) + 1
                )
                tour = np.random.choice(
                    unpicked_cities,
                    size=min(max_tour_size, len(unpicked_cities)),
                    replace=False,
                )
                unpicked_cities = np.array(
                    [city for city in unpicked_cities if city not in tour]
                )
                tours.append(tour)

            tours.append(unpicked_cities)
            np.random.shuffle(tours)
            self.__tours = tours

        result = self.__fn(self.__tours)
        return result


class InstanceTransformer:
    __fn: t.Callable[[t.List[t.List[City]]], TspResult]
    __cities: t.List[City]

    def __init__(
        self,
        fn: t.Callable[[t.List[t.List[City]]], TspResult],
        cities: t.List[City],
    ):
        self.__fn = fn
        self.__cities = cities

    def __call__(self, salesmen_routes: t.List[npt.NDArray[np.int64]]) -> TspResult:
        salesmen_routes_copy = [
            np.array([self.__cities[city] for city in salesman_route], dtype=City)
            for salesman_route in salesmen_routes
        ]

        result = self.__fn(salesmen_routes_copy)
        return result


class InstanceAugmenter:
    __fn: t.Callable[[t.List[t.List[City]]], TspResult]
    __home_city: City

    def __init__(
        self, fn: t.Callable[[t.List[t.List[City]]], TspResult], home_city: City
    ):
        self.__fn = fn
        self.__home_city = home_city

    def __call__(self, salesmen_routes: t.List[t.List[City]]) -> TspResult:
        salesmen_routes_copy = [
            np.concatenate(([self.__home_city], salesmen_route))
            for salesmen_route in salesmen_routes
        ]

        result = self.__fn(salesmen_routes_copy)
        return result


class MinMaxMultipleTSP:
    __costs: t.List[t.List[float]]

    def __init__(self, costs: t.List[t.List[float]]) -> None:
        self.__costs = costs

    def __call__(self, salesmen_routes: t.List[t.List[City]]) -> TspResult:
        result = TspResult()

        for salesman_route in salesmen_routes:
            cost = self.__tsp(salesman_route)

            result.optimal_min_max.min = min(result.optimal_min_max.min, cost)
            result.optimal_min_max.max = max(result.optimal_min_max.max, cost)
            result.optimal_cost += cost

        return result

    def __tsp(self, salesman_route: t.List[City]) -> float:
        total_cost = 0

        for i in range(len(salesman_route) - 1):
            total_cost += self.__costs[salesman_route[i]][salesman_route[i + 1]]

        return total_cost


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
                [city.identifier - 1 for city in self.__function_definition.values[1:]],
                size=len(self.__function_definition.values) - 1,
                replace=False,
            )
            for _ in range(self.__population_size)
        ]
