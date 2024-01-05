import time
import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.apso.algorithm import AdaptiveParticleSwarmOptimisation
from functions.combinatorial.definition import CombinatorialFunctionDefinition
from util.sort import maximise, minimise


class MeteredAdaptiveParticleSwarmOptimisation(AdaptiveParticleSwarmOptimisation):
    __metrics_runtime: t.List[t.Tuple[np.uint64, np.uint64]]
    __metrics_values: t.List[t.Tuple[np.uint64, t.List[np.int64]]]
    __metrics_fitness: t.List[t.Tuple[np.uint64, np.float32]]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], npt.NDArray[np.float32]],
        fitness_function: t.Callable[[np.float32], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        criteria_function: t.Callable[[t.List[np.int64], np.float32, np.uint64], bool],
        two_opt_operator_probability: float = 0.3333,
        path_linker_operator_probability: float = 0.3333,
        swap_operator_probability: float = 0.3334,
        debug: bool = False,
    ) -> None:
        super().__init__(
            generate_initial_population=generate_initial_population,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
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

        return MeteredAdaptiveParticleSwarmOptimisation(
            generate_initial_population=generate_initial_population,
            fitness_function=function_definition.function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            two_opt_operator_probability=two_opt_operator_probability,
            path_linker_operator_probability=path_linker_operator_probability,
            swap_operator_probability=swap_operator_probability,
            debug=debug,
        )

    def run(self) -> t.Tuple[t.List[np.int64], np.float32, np.uint64]:
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
    def metrics_runtime(self) -> t.List[t.Tuple[np.uint64, np.uint64]]:
        return self.__metrics_runtime

    @property
    def metrics_values(self) -> t.List[t.Tuple[np.uint64, t.List[np.int64]]]:
        return self.__metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self.__metrics_fitness
