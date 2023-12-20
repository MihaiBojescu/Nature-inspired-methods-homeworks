from functions.combinatorial.definition import CombinatorialFunctionDefinition
from functions.combinatorial.tsp.parser import Parser
from functions.combinatorial.tsp.common import (
    augment_instances,
    min_max_multiple_tsp,
    segment_instances,
)

__parser_result = Parser().parse("./src/functions/combinatorial/tsp/eil51.tsp")


def make_eil51(dimensions: int) -> CombinatorialFunctionDefinition:
    function = min_max_multiple_tsp
    function = augment_instances(fn=function, home_city=__parser_result.coordinates[0])
    function = segment_instances(fn=function, dimensions=dimensions)

    return CombinatorialFunctionDefinition(
        name=__parser_result.name,
        description=__parser_result.description,
        function=function,
        target="minimise",
        values=__parser_result.coordinates,
    )
