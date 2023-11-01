#!/usr/bin/env python3
from functions.griewangk import run_griewangk
from functions.h1p import run_h1p
from functions.michalewicz import run_michalewicz
from functions.rastringin import run_rastrigin
from functions.rosenbrock_valley import run_rosenbrock_valley


def main():
    run_rastrigin(2)
    run_rastrigin(30)
    run_rastrigin(100)

    run_griewangk(2)
    run_griewangk(30)
    run_griewangk(100)

    run_rosenbrock_valley(2)
    run_rosenbrock_valley(30)
    run_rosenbrock_valley(100)

    run_michalewicz(2)
    run_michalewicz(30)
    run_michalewicz(100)

    run_h1p()


if __name__ == "__main__":
    main()
