import numpy as np
import numpy.typing as npt


module = "H1'"


def h1p(x: npt.NDArray[np.float32]):
    return x**3 - 60 * (x**2) + 900 * x + 100


pso_parameters = {
    "generate_initial_population": lambda: [
        [np.random.uniform(low=0, high=31.0)] for _ in range(100)
    ],
    "fitness_function": h1p,
    "fitness_compare_function": lambda a, b: a > b,
    "criteria_function": lambda _population, best_fitness, generation: 4099
    < best_fitness
    < 4100,
    "inertia_bias": 0.3,
    "personal_best_position_bias": 0.7,
    "team_best_position_bias": 0.8,
    "random_jitter_bias": 0.02,
}
