import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.apso.individual import Individual
from algorithms.base.algorithm import BaseAlgorithm
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from util.sort import maximise, minimise, quicksort


class AdaptiveParticleSwarmOptimisation(BaseAlgorithm):
    _population: t.List[Individual]
    _generation: np.uint64
    _best_individual: Individual
    _fitness_compare_function: t.Callable[[np.float32, np.float32], bool]
    _criteria_function: t.Callable[[t.List[np.int64], np.float32, np.uint64], bool]

    _two_opt_operator_probability: float
    _path_linker_operator_probability: float
    _swap_operator_probability: float

    __debug: bool

    def __init__(
        self,
        generate_initial_population: t.Callable[[], npt.NDArray[np.float32]],
        fitness_function: t.Callable[[t.List[np.int64]], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        criteria_function: t.Callable[[t.List[np.int64], np.float32, np.uint64], bool],
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

        self._population = [
            Individual(
                initial_position=position,
                fitness_function=fitness_function,
                fitness_compare_function=fitness_compare_function,
                two_opt_operator_probability=two_opt_operator_probability,
                path_linker_operator_probability=path_linker_operator_probability,
                swap_operator_probability=swap_operator_probability,
            )
            for position in generate_initial_population()
        ]
        self._generation = np.uint64(0)
        self._fitness_compare_function = fitness_compare_function
        self._criteria_function = criteria_function

        self.__debug = debug

        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(a.fitness, b.fitness),
        )
        self._best_individual = self._population[0]

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
        dimensions: int = 1,
        population_size: int = 100,
        generate_initial_population: t.Union[
            t.Literal["auto"],
            t.Callable[[], t.List[np.int64]],
        ] = "auto",
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[t.List[np.int64], np.float32, np.uint64], bool],
        ] = "auto",
        two_opt_operator_probability: float = 0.3333,
        path_linker_operator_probability: float = 0.3333,
        swap_operator_probability: float = 0.3334,
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
        criteria_function = criteria_function if criteria_function != "auto" else default_criteria_function
        fitness_compare_function = maximise if function_definition.target == "maximise" else minimise

        return AdaptiveParticleSwarmOptimisation(
            generate_initial_population=generate_initial_population,
            fitness_function=function_definition.function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            two_opt_operator_probability=two_opt_operator_probability,
            path_linker_operator_probability=path_linker_operator_probability,
            swap_operator_probability=swap_operator_probability,
            debug=debug,
        )

    @property
    def name(self) -> str:
        return "APSO algorithm"

    def run(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        while not self._criteria_function(
            self._best_individual.position, self._best_individual.fitness, self._generation
        ):
            self.step()

        return self._best_individual.position, self._best_individual.fitness, self._generation

    def step(self) -> t.Tuple[t.List[np.int64], np.float32, np.uint64]:
        self._print()

        for individual in self._population:
            individual.update()

        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(a.fitness, b.fitness),
        )
        self._best_individual = self._population[0]
        self._generation += 1

        return self._best_individual.position, self._best_individual.fitness, self._generation

    def _print(self) -> None:
        if not self.__debug:
            return

        print(
            f"Adaptive particle swarm optimisation algorithm generation {self._generation}: {self._best_individual.fitness}"
        )
