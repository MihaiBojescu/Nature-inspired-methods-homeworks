import copy
import random
import typing as t
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from algorithms.base.algorithm import BaseAlgorithm
from algorithms.apso.individual import Individual
from algorithms.apso.operators import BasePathLinkerOperator, BaseSwapOperator, BaseTwoOptOperator
from util.sort import maximise, minimise, quicksort

T = t.TypeVar("T")
U = t.TypeVar("U")


class AdaptiveParticleSwarmOptimisation(BaseAlgorithm[T, U]):
    __population: t.List[Individual[T, U]]
    _best_individual: Individual
    __fitness_compare_function: t.Callable[[U, U], bool]
    _criteria_function: t.Callable[[T, U, int], bool]

    __two_opt_operator: BaseTwoOptOperator[T, U]
    __path_linker_operator: BasePathLinkerOperator[T, U]
    __swap_operator: BaseSwapOperator[T, U]
    __two_opt_operator_probability: float
    __path_linker_operator_probability: float
    __swap_operator_probability: float

    _generation: int

    __debug: bool

    def __init__(
        self,
        generate_initial_population: t.Callable[[], t.List[T]],
        fitness_function: t.Callable[[T], U],
        fitness_compare_function: t.Callable[[U, U], bool],
        criteria_function: t.Callable[[T, U, int], bool],
        two_opt_operator: BaseTwoOptOperator[T, U],
        path_linker_operator: BasePathLinkerOperator[T, U],
        swap_operator: BaseSwapOperator[T, U],
        two_opt_operator_probability: float,
        path_linker_operator_probability: float,
        swap_operator_probability: float,
        debug: bool = False,
    ) -> None:
        self.__check_operator_probabilities(
            two_opt_operator_probability=two_opt_operator_probability,
            path_linker_operator_probability=path_linker_operator_probability,
            swap_operator_probability=swap_operator_probability,
        )

        self.__population = [
            Individual(
                initial_position=position,
                fitness_function=fitness_function,
            )
            for position in generate_initial_population()
        ]
        self.__fitness_compare_function = fitness_compare_function
        self._criteria_function = criteria_function

        self.__two_opt_operator = two_opt_operator
        self.__path_linker_operator = path_linker_operator
        self.__swap_operator = swap_operator
        self.__two_opt_operator_probability = two_opt_operator_probability
        self.__path_linker_operator_probability = path_linker_operator_probability
        self.__swap_operator_probability = swap_operator_probability

        self._generation = 0
        self.__debug = debug

        self.__population = quicksort(
            data=self.__population,
            comparator=lambda a, b: self.__fitness_compare_function(a.fitness, b.fitness),
        )
        self._best_individual = self.__population[0]

    def __check_operator_probabilities(
        self,
        two_opt_operator_probability: float,
        path_linker_operator_probability: float,
        swap_operator_probability: float,
    ):
        if two_opt_operator_probability + path_linker_operator_probability + swap_operator_probability != 1.0:
            raise RuntimeError("Operator probabilities do not sum up to 1.0f")

    @staticmethod
    def from_function_definition(
        function_definition: CombinatorialFunctionDefinition,
        population_size: int = 100,
        generate_initial_population: t.Union[
            t.Literal["auto"],
            t.Callable[[], T],
        ] = "auto",
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[T, U, int], bool],
        ] = "auto",
        two_opt_operator: t.Union[t.Literal["auto"], BaseTwoOptOperator[T, U]] = "auto",
        path_linker_operator: t.Union[t.Literal["auto"], BasePathLinkerOperator[T, U]] = "auto",
        swap_operator: t.Union[t.Literal["auto"], BaseSwapOperator[T, U]] = "auto",
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

        return AdaptiveParticleSwarmOptimisation(
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

    @property
    def name(self) -> str:
        return "APSO algorithm"

    def run(self) -> t.Tuple[T, U, int]:
        while not self._criteria_function(
            self._best_individual.position, self._best_individual.fitness, self._generation
        ):
            self.step()

        return self._best_individual.position, self._best_individual.fitness, self._generation

    def step(self) -> t.Tuple[T, U, int]:
        self._print()

        for i, individual in enumerate(self.__population):
            self.__population[i] = self.__update_individual(individual)

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

    def __update_individual(self, individual: Individual[T, U]):
        match random.choices(
            [1, 2, 3],
            weights=[
                self.__two_opt_operator_probability,
                self.__path_linker_operator_probability,
                self.__swap_operator_probability,
            ],
            k=1,
        )[0]:
            case 1:
                return self.__two_opt_operator.run(individual)
            case 2:
                return self.__path_linker_operator.run(individual)
            case 3:
                return self.__swap_operator.run(individual)

    def _print(self) -> None:
        if not self.__debug:
            return

        print(
            f"Adaptive particle swarm optimisation algorithm generation {self._generation}: {self._best_individual.fitness}"
        )
