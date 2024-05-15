import copy
import random
import typing as t
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from algorithms.base.algorithm import BaseAlgorithm
from algorithms.cpso.individual import Individual
from util.sort import maximise, minimise, quicksort

T = t.TypeVar("T")


class CombinatorialParticleSwarmOptimisation(BaseAlgorithm[T]):
    __population: t.List[Individual]
    _generation: int
    _best_individual: Individual
    __fitness_compare_function: t.Callable[[float, float], bool]
    __criteria_function: t.Callable[[T, float, int], bool]

    __debug: bool

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[T]],
        fitness_function: t.Callable[[float], float],
        fitness_compare_function: t.Callable[[float, float], bool],
        criteria_function: t.Callable[[T, float, int], bool],
        inertia_weight: float,
        cognitive_parameter: float,
        social_parameter: float,
        intensification_parameter: float,
        debug: bool = False,
    ) -> None:
        self.__population = [
            Individual(
                initial_position=position,
                inertia_weight=inertia_weight,
                cognitive_parameter=cognitive_parameter,
                social_parameter=social_parameter,
                intensification_parameter=intensification_parameter,
                fitness_function=fitness_function,
                fitness_compare_function=fitness_compare_function,
            )
            for position in generate_initial_population()
        ]
        self.__fitness_compare_function = fitness_compare_function
        self.__criteria_function = criteria_function

        self._generation = 0
        self.__debug = debug

        self.__population = quicksort(
            data=self.__population,
            comparator=lambda a, b: self.__fitness_compare_function(a.fitness, b.fitness),
        )
        self._best_individual = self.__population[0]

    @staticmethod
    def from_function_definition(
        function_definition: CombinatorialFunctionDefinition,
        population_size: int = 100,
        generate_initial_population: t.Union[
            t.Literal["auto"],
            t.Callable[[], t.List[T]],
        ] = "auto",
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[t.List[T], float, int], bool],
        ] = "auto",
        inertia_bias: float = 0.4,
        personal_best_position_bias: float = 0.7,
        team_best_position_bias: float = 0.6,
        intensification_parameter: float = 0.2,
        debug: bool = False,
    ) -> t.Self:
        def default_criteria_function(_values, _fitness, generation):
            return generation > 100

        def default_generate_initial_population():
            return [
                random.sample(
                    function_definition.values,
                    k=len(function_definition.values),
                )
                for _ in range(population_size)
            ]

        generate_initial_population = (
            generate_initial_population
            if generate_initial_population != "auto"
            else default_generate_initial_population
        )
        criteria_function = criteria_function if criteria_function != "auto" else default_criteria_function
        fitness_compare_function = maximise if function_definition.target == "maximise" else minimise

        return CombinatorialParticleSwarmOptimisation(
            generate_initial_population=generate_initial_population,
            fitness_function=function_definition.function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            inertia_weight=inertia_bias,
            cognitive_parameter=personal_best_position_bias,
            social_parameter=team_best_position_bias,
            intensification_parameter=intensification_parameter,
            debug=debug,
        )

    @property
    def name(self) -> str:
        return "C-PSO algorithm"

    def run(self) -> t.Tuple[T, float, int]:
        while not self.__criteria_function(
            self._best_individual.position, self._best_individual.fitness, self._generation
        ):
            self.step()

        return self._best_individual.position, self._best_individual.fitness, self._generation

    def step(self) -> t.Tuple[T, float, int]:
        self._print()

        for individual in self.__population:
            individual.update(team_best_position=self._best_individual.personal_best_position)

        self.__population = quicksort(
            data=self.__population,
            comparator=lambda a, b: self.__fitness_compare_function(a.fitness, b.fitness),
        )
        self._best_individual = (
            copy.deepcopy(self.__population[0])
            if self.__fitness_compare_function(self.__population[0].fitness, self._best_individual.fitness)
            else self._best_individual
        )
        self._generation += 1

        return self._best_individual.position, self._best_individual.fitness, self._generation

    def _print(self) -> None:
        if not self.__debug:
            return

        print(
            f"Combinatorial particle swarm optimisation algorithm generation {self._generation}: {self._best_individual.fitness}"
        )
