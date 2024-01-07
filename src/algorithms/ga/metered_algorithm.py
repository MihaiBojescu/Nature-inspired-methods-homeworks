import random
import time
import typing as t
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from algorithms.ga.algorithm import GeneticAlgorithm
from algorithms.ga.operators import BaseCrossoverOperator, BaseMutationOperator, BaseSelectionOperator
from util.sort import maximise, minimise

T = t.TypeVar("T")


class MeteredGeneticAlgorithm(GeneticAlgorithm[T]):
    __metrics_runtime: t.List[t.Tuple[int, int]]
    __metrics_values: t.List[t.Tuple[int, T]]
    __metrics_fitness: t.List[t.Tuple[int, float]]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[T]],
        fitness_function: t.Callable[[T], float],
        fitness_compare_function: t.Callable[[float, float], bool],
        criteria_function: t.Callable[[T, float, int], bool],
        selection_operator: BaseSelectionOperator[T],
        crossover_operator: BaseCrossoverOperator[T],
        mutation_operator: BaseMutationOperator[T],
        mutation_chance: float,
        debug: bool = False,
    ) -> None:
        super().__init__(
            generate_initial_population=generate_initial_population,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            mutation_chance=mutation_chance,
            debug=debug,
        )

        self.__metrics_runtime = []
        self.__metrics_values = []
        self.__metrics_fitness = []

    @staticmethod
    def from_function_definition(
        function_definition: CombinatorialFunctionDefinition,
        population_size: int = 100,
        generations: int = 100,
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[T, float, int], bool],
        ] = "auto",
        selection_operator: t.Union[
            t.Literal["auto"],
            BaseSelectionOperator,
        ] = "auto",
        crossover_operator: t.Union[
            t.Literal["auto"],
            BaseCrossoverOperator,
        ] = "auto",
        mutation_operator: t.Union[
            t.Literal["auto"],
            BaseMutationOperator,
        ] = "auto",
        mutation_chance: float = 0.02,
        debug: bool = False,
    ) -> t.Self:
        cached_min_best_result = function_definition.best_result - 0.05
        cached_max_best_result = function_definition.best_result + 0.05
        criteria_function = (
            criteria_function
            if criteria_function != "auto"
            else lambda _values, fitness, generation: generation > generations
            or cached_min_best_result < fitness < cached_max_best_result
        )

        selection_operator = selection_operator if selection_operator != "auto" else BaseSelectionOperator()
        crossover_operator = crossover_operator if crossover_operator != "auto" else BaseCrossoverOperator()
        mutation_operator = mutation_operator if mutation_operator != "auto" else BaseMutationOperator()

        return MeteredGeneticAlgorithm(
            generate_initial_population=lambda: [
                random.sample(function_definition.values, k=len(function_definition.values))
                for _ in range(population_size)
            ],
            fitness_function=function_definition.function,
            fitness_compare_function=maximise if function_definition.target == "maximise" else minimise,
            criteria_function=criteria_function,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            mutation_chance=mutation_chance,
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
    def metrics_values(self) -> t.List[t.Tuple[int, T]]:
        return self.__metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[int, float]]:
        return self.__metrics_fitness
