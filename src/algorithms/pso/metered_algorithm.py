import time
import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.pso.algorithm import ParticleSwarmOptimisation
from functions.definition import FunctionDefinition
from util.sort import maximise, minimise


class MeteredParticleSwarmOptimisation(ParticleSwarmOptimisation):
    __metrics_runtime: t.List[t.Tuple[np.uint64, np.uint64]]
    __metrics_values: t.List[t.Tuple[np.uint64, np.float32]]
    __metrics_fitness: t.List[t.Tuple[np.uint64, np.float32]]

    def __init__(
        self,
        generate_initial_population: t.Callable[[], npt.NDArray[np.float32]],
        fitness_function: t.Callable[[np.float32], np.float32],
        fitness_compare_function: t.Callable[[np.float32, np.float32], bool],
        criteria_function: t.Callable[
            [t.List[np.float32], np.float32, np.uint64], bool
        ],
        inertia_weight: np.float32,
        cognitive_parameter: np.float32,
        social_parameter: np.float32,
        random_jitter_parameter: np.float32,
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
            random_jitter_parameter=random_jitter_parameter,
            debug=debug,
        )

        self.__metrics_runtime = []
        self.__metrics_values = []
        self.__metrics_fitness = []

    @staticmethod
    def from_function_definition(
        function_definition: FunctionDefinition,
        dimensions: int = 1,
        population_size: int = 100,
        generations: int = 100,
        criteria_function: t.Union[
            t.Literal["auto"],
            t.Callable[[t.List[np.float32], np.float32, np.uint64], bool],
        ] = "auto",
        inertia_weight: float = 0.4,
        cognitive_parameter: float = 0.7,
        social_parameter: float = 0.6,
        random_jitter_parameter: float = 0.02,
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

        return MeteredParticleSwarmOptimisation(
            generate_initial_population=lambda: [
                [
                    np.random.uniform(
                        low=function_definition.value_boundaries.min,
                        high=function_definition.value_boundaries.max,
                    )
                    for _ in range(dimensions)
                ]
                for _ in range(population_size)
            ],
            fitness_function=function_definition.function,
            fitness_compare_function=maximise
            if function_definition.target == "maximise"
            else minimise,
            criteria_function=criteria_function,
            inertia_weight=inertia_weight,
            cognitive_parameter=cognitive_parameter,
            social_parameter=social_parameter,
            random_jitter_parameter=random_jitter_parameter,
            debug=debug,
        )

    def run(self) -> t.Tuple[np.float32, np.float32, np.uint64]:
        then = time.time_ns()
        best_individual = self._population[0]

        while not self._criteria_function(
            best_individual.position, best_individual.fitness, self._generation
        ):
            self.step()

            now = time.time_ns()
            self.__metrics_runtime.append((self._generation, now - then))

            for individual in self._population:
                self.__metrics_values.append((self._generation, individual.position))
                self.__metrics_fitness.append((self._generation, individual.fitness))

        best_individual = self._population[0]

        return best_individual.position, best_individual.fitness, self._generation

    @property
    def metrics_runtime(self) -> t.List[t.Tuple[np.uint64, np.uint64]]:
        return self.__metrics_runtime

    @property
    def metrics_values(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self.__metrics_values

    @property
    def metrics_fitness(self) -> t.List[t.Tuple[np.uint64, np.float32]]:
        return self.__metrics_fitness
