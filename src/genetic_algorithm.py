import typing as t
import numpy as np
import numpy.typing as npt


class GeneticAlgorithm:
    population: npt.NDArray[np.float32]
    fitness_function: t.Callable[[np.float32], np.float32]
    criteria_function: t.Callable[[t.List[t.Tuple[np.float32, np.float32]]], np.float32]
    selection_function: t.Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
    crossover_function: t.Callable[[np.float32, np.float32], np.float32]
    mutation_chance: np.float16

    def __init__(
        self,
        generate_initial_population: t.Callable[[], npt.NDArray[np.float32]],
        fitness_function: t.Callable[[np.float32], np.float32],
        criteria_function: t.Callable[[npt.NDArray[np.float32]], bool],
        selection_function: t.Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
        crossover_function: t.Callable[[np.float32, np.float32], np.float32],
        mutation_chance: np.float16
    ) -> None:
        self.population = generate_initial_population()
        self.fitness_function = fitness_function
        self.criteria_function = criteria_function
        self.selection_function = selection_function
        self.crossover_function = crossover_function
        self.mutation_chance = mutation_chance

    def run(self):
        population_fitness = [(individual, self.fitness_function(individual)) for individual in self.population]

        while self.criteria_function(population_fitness):
            break

