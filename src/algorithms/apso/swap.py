import typing as t
import numpy as np

class Swap:
    def run(self, values: t.List[np.int64]):
        [a, b] = np.random.choice(list(range(len(values))))
        values[a], values[b] = values[b], values[a]

        return values
