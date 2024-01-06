import typing as t
import numpy as np
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from algorithms.base.algorithm import BaseAlgorithm
from algorithms.ga.individual import Individual
from algorithms.ga.encoder import BaseEncoder
from algorithms.ga.operators import BaseCrossoverOperator, BaseMutationOperator, BaseSelectionOperator
from util.sort import maximise, minimise, quicksort

T = t.TypeVar("T")


class GeneticAlgorithm(BaseAlgorithm[T]):
    __population: t.List[Individual[T]]
    _best_individual: Individual[T]
    __fitness_compare_function: t.Callable[[np.float32, np.float32], bool]
    __criteria_function: t.Callable[[T, np.float32, np.uint64], bool]

    __selection_operator: BaseSelectionOperator[T]
    __crossover_operator: BaseCrossoverOperator[T]
    __mutation_operator: BaseMutationOperator[T]

    _generation: np.uint64

    __debug: bool

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[T]],
        fitness_function: t.Callable[[T], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        criteria_function: t.Callable[[T, np.float32, np.uint64], bool],
        selection_operator: BaseSelectionOperator[T],
        crossover_operator: BaseCrossoverOperator[T],
        mutation_operator: BaseMutationOperator[T],
        mutation_chance: np.float16,
        debug: bool = False,
    ) -> None:
        self.__population = [
            Individual(genes=genes, fitness_function=fitness_function) for genes in generate_initial_population()
        ]
        self.__fitness_function = fitness_function
        self.__fitness_compare_function = fitness_compare_function
        self.__selection_function = selection_operator
        self.__criteria_function = criteria_function

        self.__selection_operator = selection_operator
        self.__crossover_operator = crossover_operator
        self.__mutation_operator = mutation_operator
        self.__mutation_chance = mutation_chance

        self._generation = np.uint64(0)

        self.__debug = debug

        self.__population = quicksort(
            data=self.__population,
            comparator=lambda a, b: self.__fitness_compare_function(a, b),
        )
        self._best_individual = self.__population[0]

    @staticmethod
    def from_function_definition(
        function_definition: CombinatorialFunctionDefinition,
        population_size: int = 100,
        generations: int = 100,
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[T, np.float32, np.uint64], bool],
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
        mutation_chance: np.float16 = 0.02,
        debug: bool = False,
    ):
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

        return GeneticAlgorithm(
            generate_initial_population=lambda: [
                np.random.choice(function_definition.values, size=len(function_definition.values), replace=False)
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

    @property
    def name(self) -> str:
        return "Genetic algorithm"

    def run(self) -> t.Tuple[T, np.float32, np.uint64]:
        while not self.__criteria_function(
            self._best_individual.genes, self._best_individual.fitness, self._generation
        ):
            self.step()

        return self._best_individual.genes, self._best_individual.fitness, self._generation

    def step(self) -> t.Tuple[T, np.float32, np.uint64]:
        self.__print()

        next_generation = []
        selected_population = self.__selection_operator.run(self.__population)

        for index in range(0, len(selected_population) - 1, 2):
            parent_1 = selected_population[index]
            parent_2 = selected_population[index + 1]

            child_1, child_2 = self.__crossover_operator.run(parent_1, parent_2)

            child_1 = self.__mutation_operator.run(child_1)
            child_2 = self.__mutation_operator.run(child_2)

            next_generation.extend([child_1, child_2])

        self.__population = next_generation
        self.__population = quicksort(
            data=self.__population,
            comparator=lambda a, b: self.__fitness_compare_function(a.fitness, b.fitness),
        )
        self._best_individual = self.__population[0]
        self._generation += 1

        return self._best_individual.decode(), self._best_individual.fitness, self._generation

    def __print(self) -> None:
        if not self.__debug:
            return

        print(f"Genetic algorithm generation: {self._generation}: {self._best_individual.fitness}")
