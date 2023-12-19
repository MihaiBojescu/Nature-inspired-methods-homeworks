import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.cpso.individual import Individual
from algorithms.base.algorithm import BaseAlgorithm
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from util.sort import maximise, minimise, quicksort

T = t.TypeVar("T")
U = t.TypeVar("U")


class CombinatorialParticleSwarmOptimisation(BaseAlgorithm):
    _population: t.List[Individual]
    _generation: np.uint64
    _fitness_compare_function: t.Callable[[np.float32, np.float32], bool]
    _criteria_function: t.Callable[[t.List[np.float32], np.float32, np.uint64], bool]

    __debug: bool

    def __init__(
        self,
        generate_initial_population: t.Callable[[], npt.NDArray[np.float32]],
        fitness_function: t.Callable[[np.float32], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        criteria_function: t.Callable[[t.List[T], np.float32, np.uint64], bool],
        inertia_weight: np.float32,
        cognitive_parameter: np.float32,
        social_parameter: np.float32,
        intensification_parameter: np.float32,
        debug: bool = False,
    ) -> None:
        self._population = [
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
        self._generation = np.uint64(0)
        self._fitness_compare_function = fitness_compare_function
        self._criteria_function = criteria_function

        self.__debug = debug

        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )

    @staticmethod
    def from_function_definition(
        function_definition: CombinatorialFunctionDefinition,
        dimensions: int = 1,
        population_size: int = 100,
        generate_initial_population: t.Union[
            t.Literal["auto"],
            t.Callable[[], t.List[T]],
        ] = "auto",
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[t.List[T], np.float32, np.uint64], bool],
        ] = "auto",
        inertia_bias: float = 0.4,
        personal_best_position_bias: float = 0.7,
        team_best_position_bias: float = 0.6,
        intensification_parameter: np.float32 = 0.2,
        debug: bool = False,
    ):
        def default_criteria_function(_values, _fitness, generation):
            return generation > 100

        def default_generate_initial_population():
            return [
                [
                    np.random.choice(
                        function_definition.values,
                        size=len(function_definition.values),
                        replace=False,
                    )
                    for _ in range(dimensions)
                ]
                for _ in range(population_size)
            ]

        generate_initial_population = (
            generate_initial_population
            if generate_initial_population != "auto"
            else default_generate_initial_population
        )
        criteria_function = (
            criteria_function
            if criteria_function != "auto"
            else default_criteria_function
        )
        fitness_compare_function = (
            maximise if function_definition.target == "maximise" else minimise
        )

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

    def run(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        best_individual = self._population[0]

        while not self._criteria_function(
            best_individual.position, best_individual.fitness, self._generation
        ):
            self.step()

        best_individual = self._population[0]

        return best_individual.position, best_individual.fitness, self._generation

    def step(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        self._print(self._generation)
        best_individual = self._population[0]

        for individual in self._population:
            individual.update(team_best_position=best_individual.personal_best_position)

        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )
        best_individual = self._population[0]

        self._generation += 1

        return best_individual.position, best_individual.fitness, self._generation

    def _print(self, generation: int) -> None:
        if not self.__debug:
            return

        best_individual = self._population[0]
        print(
            f"Combinatorial particle swarm optimisation algorithm generation {generation}: {best_individual.fitness}"
        )
