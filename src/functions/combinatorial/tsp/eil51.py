from functions.combinatorial.definition import CombinatorialFunctionDefinition
from functions.combinatorial.tsp.parser import Parser
from functions.combinatorial.tsp.common import (
    CostCalculator,
    InstanceAugmenter,
    InstanceTransformer,
    Segmenter,
    MinMaxMultipleTSP,
)

__parser_result = Parser().parse("./src/functions/combinatorial/tsp/eil51.tsp")


def make_eil51(dimensions: int) -> CombinatorialFunctionDefinition:
    cost_calculator = CostCalculator()
    segmenter = Segmenter(cities=len(__parser_result.coordinates) - 1, salesmen=dimensions)
    costs = cost_calculator(__parser_result.coordinates)

    function = MinMaxMultipleTSP(costs=costs)
    function = InstanceAugmenter(fn=function, home_city=__parser_result.coordinates[0])
    function = InstanceTransformer(fn=function, cities=__parser_result.coordinates)

    return CombinatorialFunctionDefinition(
        name=__parser_result.name,
        description=__parser_result.description,
        function=function,
        target="minimise",
        values=__parser_result.coordinates,
        segmentation=segmenter.segments,
        costs=costs,
    )
