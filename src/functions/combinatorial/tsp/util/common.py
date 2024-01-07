from dataclasses import dataclass, field
import math
import random
import typing as t
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from functions.combinatorial.tsp.util.data import City


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
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


@dataclass
class MTSPResultOptimalMinMax:
    min: float = float("inf")
    max: float = float("-inf")


@dataclass
class MTSPResult:
    optimal_min_max: MTSPResultOptimalMinMax = field(default_factory=MTSPResultOptimalMinMax)
    optimal_cost: float = 0

    def __add__(self, other):
        return self.optimal_cost + other.optimal_cost

    def __gt__(self, other):
        return self.optimal_cost > other.optimal_cost

    def __eq__(self, other):
        return self.optimal_cost == other.optimal_cost

    def __lt__(self, other):
        return self.optimal_cost < other.optimal_cost


class InstanceTransformer:
    __fn: t.Callable[[t.List[t.List[City]]], MTSPResult]
    __cities: t.List[City]

    def __init__(
        self,
        fn: t.Callable[[t.List[t.List[City]]], MTSPResult],
        cities: t.List[City],
    ):
        self.__fn = fn
        self.__cities = cities

    def __call__(self, salesmen_routes: t.List[t.List[int]]) -> MTSPResult:
        salesmen_routes_copy = [[self.__cities[city] for city in salesman_route] for salesman_route in salesmen_routes]

        result = self.__fn(salesmen_routes_copy)
        return result


class InstanceAugmenter:
    __fn: t.Callable[[t.List[t.List[City]]], MTSPResult]
    __home_city: City

    def __init__(self, fn: t.Callable[[t.List[t.List[City]]], MTSPResult], home_city: City):
        self.__fn = fn
        self.__home_city = home_city

    def __call__(self, salesmen_routes: t.List[t.List[City]]) -> MTSPResult:
        salesmen_routes_copy = [[self.__home_city] + salesmen_route for salesmen_route in salesmen_routes]

        result = self.__fn(salesmen_routes_copy)
        return result


class Segmenter:
    __cities: int
    __salesmen: int
    __segments: t.List[int]

    def __init__(
        self,
        cities: int,
        salesmen: int,
    ):
        if salesmen < 1:
            raise IndexError("Salesmen < 1 in segment function")

        self.__cities = cities
        self.__salesmen = salesmen
        self.__segments = self.__perform_segmentation()

    def __perform_segmentation(self) -> t.List[t.List[int]]:
        segments = []
        unpicked_cities = self.__cities

        for salesman in range(self.__salesmen, 1, -1):
            segment = math.floor(((unpicked_cities - 1 + salesman) / salesman) + 1)
            unpicked_cities -= segment
            segments.append(segment)

        segments.append(unpicked_cities)
        random.shuffle(segments)
        return segments

    @property
    def segments(self) -> t.List[int]:
        return self.__segments


class MinMaxMultipleTSP:
    __costs: t.List[t.List[float]]

    def __init__(self, costs: t.List[t.List[float]]) -> None:
        self.__costs = costs

    def __call__(self, salesmen_routes: t.List[t.List[City]]) -> MTSPResult:
        result = MTSPResult()

        for salesman_route in salesmen_routes:
            cost = self.__tsp(salesman_route)

            result.optimal_min_max.min = min(result.optimal_min_max.min, cost)
            result.optimal_min_max.max = max(result.optimal_min_max.max, cost)
            result.optimal_cost += cost

        return result

    def __tsp(self, salesman_route: t.List[City]) -> float:
        total_cost = 0

        for i in range(len(salesman_route) - 1):
            city_a = salesman_route[i].to_index()
            city_b = salesman_route[i + 1].to_index()
            total_cost += self.__costs[city_a][city_b]

        last_city = salesman_route[-1].to_index()
        total_cost += self.__costs[last_city][0]

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
        return [self.__generate_individual() for _ in range(self.__population_size)]

    def __generate_individual(self):
        individual = random.sample(
            [city.to_index() for city in self.__function_definition.values[1:]],
            k=len(self.__function_definition.values) - 1,
        )
        individual = self.__segment(individual)

        return individual

    def __segment(self, individual: t.List[int]) -> t.List[t.List[int]]:
        result = [[None for _ in range(segment)] for segment in self.__function_definition.segmentation]
        start = 0

        for i, segment in enumerate(self.__function_definition.segmentation):
            result[i] = individual[start : start + segment]
            start += segment

        return result
