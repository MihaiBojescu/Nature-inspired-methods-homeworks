#!/usr/bin/env python3

import numpy as np
from continuous_hillclimber import ContinuousHillclimber


def main():
    algorithm = ContinuousHillclimber(
        fx=lambda x: x**3 - 60 * (x**2) + 900 * x + 100,
        interval=(np.float32(0.0), np.float32(31.0)),
        step=np.float32(0.1),
        acceleration=np.float32(0.1),
        precision=np.finfo(np.float32).eps,
        iterations=None,
    )
    for _ in range(1, 50000):
        value = algorithm.run()

        print(f"Final x: {value}")


if __name__ == "__main__":
    main()
