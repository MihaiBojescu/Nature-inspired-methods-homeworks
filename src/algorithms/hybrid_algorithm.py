import typing as t
import numpy as np
import numpy.typing as npt
from algorithms.binary_hillclimber import BinaryHillclimber
from algorithms.binary_genetic_algorithm import BinaryGeneticAlgorithm
from data.individual import DecodedIndividual, Individual
from util.sort import quicksort


class HybridAlgorithm(BinaryGeneticAlgorithm):
    _hillclimber_run_interval: np.uint32
    _hillclimber_step: np.float32
    _hillclimber_acceleration: np.float32

    def __init__(
        self,
        encode: t.Callable[[any], npt.NDArray[np.uint8]],
        decode: t.Callable[[npt.NDArray[np.uint8]], any],
        generate_initial_population: t.Callable[[], t.List[any]],
        fitness_function: t.Callable[[any], np.float32],
        fitness_compare_function: t.Callable[[any, any], bool],
        selection_function: t.Callable[
            [t.List[DecodedIndividual]],
            t.List[DecodedIndividual],
        ],
        criteria_function: t.Callable[[np.uint64, t.List[DecodedIndividual]], bool],
        crossover_points: t.List[np.uint32],
        mutation_chance: np.float16,
        hillclimber_neighbor_selection_function: t.Union[None, t.Callable[[any], bool]],
        hillclimber_run_interval: np.uint32,
        hillclimber_step: np.float32 = np.float32(0.1),
        hillclimber_acceleration: np.float32 = np.float32(0.1),
        debug: bool = False,
    ) -> None:
        super().__init__(
            encode=encode,
            decode=decode,
            generate_initial_population=generate_initial_population,
            fitness_function=fitness_function,
            fitness_compare_function=fitness_compare_function,
            criteria_function=criteria_function,
            selection_function=selection_function,
            crossover_points=crossover_points,
            mutation_chance=mutation_chance,
            debug=debug,
        )

        self._hillclimber_neighbor_selection_function = (
            hillclimber_neighbor_selection_function
        )
        self._hillclimber_run_interval = hillclimber_run_interval
        self._hillclimber_step = hillclimber_step
        self._hillclimber_acceleration = hillclimber_acceleration

    def step(self) -> t.Tuple[any, any, np.uint64]:
        self._print(self._generation)
        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )
        next_generation = []

        selected_population = self._selection_function(self.decoded_population)

        for index in range(0, len(selected_population) - 1, 2):
            parent_1 = Individual.from_decoded_individual(
                decoded_individual=selected_population[index],
                encode=self._encode,
                decode=self._decode,
                fitness_function=self._fitness_function,
            )
            parent_2 = Individual.from_decoded_individual(
                decoded_individual=selected_population[index + 1],
                encode=self._encode,
                decode=self._decode,
                fitness_function=self._fitness_function,
            )

            child_1, child_2 = self._crossover(parent_1, parent_2)

            child_1 = self._mutate(child_1)
            child_2 = self._mutate(child_2)

            next_generation.extend([child_1, child_2])

        self._population = next_generation
        self._run_hillclimber()
        self._population = quicksort(
            data=self._population,
            comparator=lambda a, b: self._fitness_compare_function(
                a.fitness, b.fitness
            ),
        )
        self._generation += 1

        best_individual = self._population[0]

        return best_individual.fitness, best_individual.value, self._generation

    def _print(self, generation: np.uint64) -> None:
        if self._debug:
            print(f"Hybrid algorithm generation: {generation}")

    def _run_hillclimber(self):
        if self._generation % self._hillclimber_run_interval != 0:
            return

        for i, individual in enumerate(self._population):
            decoded_individual = individual.value
            binary_hill_climber = BinaryHillclimber(
                encode=self._encode,
                decode=self._decode,
                generate_initial_value=lambda: decoded_individual,
                fitness_function=self._fitness_function,
                fitness_compare_function=self._fitness_compare_function,
                neighbor_selection_function=self._hillclimber_neighbor_selection_function,
                criteria_function=lambda _1, _2, _3: True,
                debug=self._debug,
            )
            _best_score, optimised_individual, _generation = binary_hill_climber.step()
            encoded_individual = self._encode(optimised_individual)

            self._population[i].genes = encoded_individual
