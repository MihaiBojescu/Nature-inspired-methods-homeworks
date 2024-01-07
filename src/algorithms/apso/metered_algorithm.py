import random
import time
import typing as t
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from algorithms.apso.algorithm import AdaptiveParticleSwarmOptimisation
from algorithms.apso.operators import BasePathLinkerOperator, BaseSwapOperator, BaseTwoOptOperator
from util.sort import maximise, minimise

T = t.TypeVar("T")


class MeteredAdaptiveParticleSwarmOptimisation(AdaptiveParticleSwarmOptimisation[T]):
    __metrics_runtime: t.List[t.Tuple[int, int]]
    __metrics_values: t.List[t.Tuple[int, T]]
    __metrics_fitness: t.List[t.Tuple[int, float]]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[T]],
        fitness_function: t.Callable[[T], float],
        fitness_compare_function: t.Callable[[float, float], bool],
        criteria_function: t.Callable[[T, float, int], bool],
        two_opt_operator: BaseTwoOptOperator[T],
        path_linker_operator: BasePathLinkerOperator[T],
        swap_operator: BaseSwapOperator[T],
        two_opt_operator_probability: float,
        path_linker_operator_probability: float,
        swap_operator_probability: float,
        debug: bool = False,
    ) -> None:
        super().__init__(
            generate_initial_population=generate_initial_population,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            two_opt_operator=two_opt_operator,
            path_linker_operator=path_linker_operator,
            swap_operator=swap_operator,
            two_opt_operator_probability=two_opt_operator_probability,
            path_linker_operator_probability=path_linker_operator_probability,
            swap_operator_probability=swap_operator_probability,
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
        two_opt_operator: t.Union[t.Literal["auto"], BaseTwoOptOperator[T]] = "auto",
        path_linker_operator: t.Union[t.Literal["auto"], BasePathLinkerOperator[T]] = "auto",
        swap_operator: t.Union[t.Literal["auto"], BaseSwapOperator[T]] = "auto",
        two_opt_operator_probability: float = 0.3333,
        path_linker_operator_probability: float = 0.3333,
        swap_operator_probability: float = 0.3334,
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
        two_opt_operator = two_opt_operator if two_opt_operator != "auto" else BaseTwoOptOperator()
        path_linker_operator = path_linker_operator if path_linker_operator != "auto" else BasePathLinkerOperator()
        swap_operator = swap_operator if swap_operator != "auto" else BaseSwapOperator()

        return MeteredAdaptiveParticleSwarmOptimisation(
            generate_initial_population=generate_initial_population,
            fitness_function=function_definition.function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            two_opt_operator=two_opt_operator,
            path_linker_operator=path_linker_operator,
            swap_operator=swap_operator,
            two_opt_operator_probability=two_opt_operator_probability,
            path_linker_operator_probability=path_linker_operator_probability,
            swap_operator_probability=swap_operator_probability,
            debug=debug,
        )

    def run(self) -> t.Tuple[T, float, int]:
        then = time.time_ns()

        while not self._criteria_function(
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
    def metrics_values(self) -> t.List[t.Tuple[int, T]]:
        return self.__metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[int, float]]:
        return self.__metrics_fitness
