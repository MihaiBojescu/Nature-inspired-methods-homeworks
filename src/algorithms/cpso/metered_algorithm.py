import random
import time
import typing as t
from algorithms.cpso.algorithm import CombinatorialParticleSwarmOptimisation
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from util.sort import maximise, minimise


T = t.TypeVar("T")


class MeteredCombinatorialParticleSwarmOptimisation(CombinatorialParticleSwarmOptimisation[T]):
    __metrics_runtime: t.List[t.Tuple[int, int]]
    __metrics_values: t.List[t.Tuple[int, t.List[int]]]
    __metrics_fitness: t.List[t.Tuple[int, float]]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[int]],
        fitness_function: t.Callable[[float], float],
        fitness_compare_function: t.Callable[[float, float], bool],
        criteria_function: t.Callable[[t.List[int], float, int], bool],
        inertia_weight: float,
        cognitive_parameter: float,
        social_parameter: float,
        intensification_parameter: float,
        debug: bool = False,
    ) -> None:
        super().__init__(
            generate_initial_population=generate_initial_population,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            inertia_weight=inertia_weight,
            cognitive_parameter=cognitive_parameter,
            social_parameter=social_parameter,
            intensification_parameter=intensification_parameter,
            debug=debug,
        )

        self.__metrics_runtime = []
        self.__metrics_values = []
        self.__metrics_fitness = []

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
            t.Callable[[T, float, int], bool],
        ] = "auto",
        inertia_weight: float = 0.4,
        cognitive_parameter: float = 0.7,
        social_parameter: float = 0.6,
        intensification_parameter: float = 0.2,
        debug: bool = False,
    ) -> t.Self:
        def default_criteria_function(_values, _fitness, generation):
            return generation > 100

        def default_generate_initial_population():
            return [
                random.choices(
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

        return MeteredCombinatorialParticleSwarmOptimisation(
            generate_initial_population=generate_initial_population,
            fitness_function=function_definition.function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            inertia_weight=inertia_weight,
            cognitive_parameter=cognitive_parameter,
            social_parameter=social_parameter,
            intensification_parameter=intensification_parameter,
            debug=debug,
        )

    def run(self) -> t.Tuple[T, float, int]:
        then = time.time_ns()

        while not self.__criteria_function(
            self._best_individual.position, self._best_individual.fitness, self._generation
        ):
            self.step()

            now = time.time_ns()
            self.__metrics_runtime.append((self._generation, now - then))
            self.__metrics_values.append((self._generation, self._best_individual.position))
            self.__metrics_fitness.append((self._generation, self._best_individual.fitness))

        return self._best_individual.position, self._best_individual.fitness, self._generation

    @property
    def metrics_runtime(self) -> t.List[t.Tuple[int, int]]:
        return self.__metrics_runtime

    @property
    def metrics_values(self) -> t.List[t.Tuple[int, float]]:
        return self.__metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[int, float]]:
        return self.__metrics_fitness
