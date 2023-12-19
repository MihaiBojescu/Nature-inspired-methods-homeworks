from functions.combinatorial.definition import CombinatorialFunctionDefinition
from functions.combinatorial.tsp.parser import Parser
from functions.combinatorial.tsp.common import augment_instances, min_max_multiple_tsp

__parser_result = Parser().parse('./src/functions/combinatorial/eil51.tsp')

eil51 = CombinatorialFunctionDefinition(
    name=__parser_result.name,
    description=__parser_result.description,
    function=augment_instances(min_max_multiple_tsp, __parser_result.coordinates[0]),
    target='minimise',
    values=__parser_result.coordinates
)