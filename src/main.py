#!/usr/bin/env python3
from functions.h1p import run_h1p
from functions.rastringin import run_rastrigin


def main():
    run_rastrigin(2)
    run_h1p()


if __name__ == "__main__":
    main()
