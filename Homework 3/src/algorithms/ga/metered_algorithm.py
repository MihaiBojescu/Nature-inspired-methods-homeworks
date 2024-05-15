import random
import time
import typing as t
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from algorithms.ga.algorithm import GeneticAlgorithm
from algorithms.ga.operators import BaseCrossoverOperator, BaseMutationOperator, BaseSelectionOperator
from util.sort import maximise, minimise

T = t.TypeVar("T")
U = t.TypeVar("U")


class MeteredGeneticAlgorithm(GeneticAlgorithm[T, U]):
    __metrics_runtime: t.List[t.Tuple[int, int]]
    __metrics_values: t.List[t.Tuple[int, T]]
    __metrics_fitness: t.List[t.Tuple[int, U]]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[T]],
        fitness_function: t.Callable[[T], U],
        fitness_compare_function: t.Callable[[U, U], bool],
        criteria_function: t.Callable[[T, U, int], bool],
        selection_operator: BaseSelectionOperator[T, U],
        crossover_operator: BaseCrossoverOperator[T, U],
        mutation_operator: BaseMutationOperator[T, U],
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
            debug=debug,
        )

        self.__metrics_runtime = []
        self.__metrics_values = []
        self.__metrics_fitness = []

    @staticmethod
    def from_function_definition(
        function_definition: CombinatorialFunctionDefinition,
        generate_initial_population: t.Union[
            t.Literal["auto"],
            t.Callable[[], t.List[T]],
        ] = "auto",
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[T, U, int], bool],
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
        debug: bool = False,
    ) -> t.Self:
        def default_generate_initial_population():
            return [
                random.sample(
                    function_definition.values,
                    k=len(function_definition.values),
                )
                for _ in range(100)
            ]

        def default_criteria_function(_values, _fitness, generation):
            return generation > 100

        generate_initial_population = (
            generate_initial_population
            if generate_initial_population != "auto"
            else default_generate_initial_population
        )
        criteria_function = criteria_function if criteria_function != "auto" else default_criteria_function
        fitness_compare_function = maximise if function_definition.target == "maximise" else minimise

        selection_operator = selection_operator if selection_operator != "auto" else BaseSelectionOperator()
        crossover_operator = crossover_operator if crossover_operator != "auto" else BaseCrossoverOperator()
        mutation_operator = mutation_operator if mutation_operator != "auto" else BaseMutationOperator()

        return MeteredGeneticAlgorithm(
            generate_initial_population=generate_initial_population,
            fitness_function=function_definition.function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            debug=debug,
        )

    def run(self) -> t.Tuple[T, U, int]:
        then = time.time_ns()

        while not self._criteria_function(
            self._best_individual.genes, self._best_individual.fitness, self._generation
        ):
            self.step()

            now = time.time_ns()
            self.__metrics_runtime.append((self._generation, now - then))
            self.__metrics_values.append((self._generation, self._best_individual.genes))
            self.__metrics_fitness.append((self._generation, self._best_individual.fitness))

        return self._best_individual.genes, self._best_individual.fitness, self._generation

    @property
    def metrics_runtime(self) -> t.List[t.Tuple[int, int]]:
        return self.__metrics_runtime

    @property
    def metrics_values(self) -> t.List[t.Tuple[int, T]]:
        return self.__metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[int, U]]:
        return self.__metrics_fitness
