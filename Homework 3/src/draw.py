#!/usr/bin/env python3
from functions.combinatorial.tsp.definitions.eil51 import make_eil51
from functions.combinatorial.tsp.definitions.berlin52 import make_berlin52
from functions.combinatorial.tsp.definitions.eil76 import make_eil76
from functions.combinatorial.tsp.definitions.rat99 import make_rat99
from util.graph import draw


def main():
    eil51 = make_eil51(dimensions=1)
    berlin52 = make_berlin52(dimensions=1)
    eil76 = make_eil76(dimensions=1)
    rat99 = make_rat99(dimensions=1)

    draw_genetic_algorithm(eil51=eil51, berlin52=berlin52, eil76=eil76, rat99=rat99)
    draw_particle_swarm_optimisation_algorithm(eil51=eil51, berlin52=berlin52, eil76=eil76, rat99=rat99)


def draw_genetic_algorithm(eil51, berlin52, eil76, rat99):
    draw(
        name="Genetic algorithm: eil51, 3 salesmen",
        cities=eil51.values.copy(),
        home_city=eil51.values[0],
        tours=[
            [47, 6, 25, 7, 30, 27, 2, 35, 34, 19, 28, 20, 33, 49, 15, 1, 21],
            [50, 46, 12, 40, 39, 18, 41, 43, 14, 44, 32, 38, 9, 29, 8, 37, 10, 31],
            [45, 11, 4, 48, 36, 16, 3, 17, 24, 13, 23, 42, 22, 5, 26],
        ],
    )
    draw(
        name="Genetic algorithm: eil51, 5 salesmen",
        cities=eil51.values.copy(),
        home_city=eil51.values[0],
        tours=[
            [31, 4, 36, 16, 3, 46, 50],
            [26, 17, 12, 40, 39, 18, 41, 43, 11, 45],
            [8, 33, 29, 9, 38, 32, 44, 14, 48, 37, 10],
            [7, 30, 25, 6, 42, 23, 24, 13, 5, 22, 47],
            [1, 15, 49, 20, 28, 19, 34, 35, 2, 27, 21],
        ],
    )

    draw(
        name="Genetic algorithm: berlin52, 3 salesmen",
        cities=berlin52.values.copy(),
        home_city=berlin52.values[0],
        tours=[
            [35, 36, 14, 42, 32, 9, 8, 7, 40, 18, 44, 2, 16, 17, 31],
            [48, 38, 39, 37, 4, 23, 47, 43, 15, 49, 19, 29, 1, 6, 41, 20, 30, 21],
            [22, 28, 46, 25, 27, 26, 12, 13, 51, 10, 50, 11, 3, 5, 24, 45, 33, 34],
        ],
    )
    draw(
        name="Genetic algorithm: berlin52, 5 salesmen",
        cities=berlin52.values.copy(),
        home_city=berlin52.values[0],
        tours=[
            [43, 25, 46, 13, 51, 12, 26, 27, 11, 24, 47, 34],
            [48, 38, 39, 37, 23, 36, 35],
            [49, 15, 45, 4, 14, 42, 7, 40, 17, 30, 21],
            [31, 2, 16, 20, 6, 1, 41, 29, 28, 19, 22],
            [44, 18, 8, 9, 32, 10, 50, 3, 5, 33],
        ],
    )

    draw(
        name="Genetic algorithm: eil76, 3 salesmen",
        cities=eil76.values.copy(),
        home_city=eil76.values[0],
        tours=[
            [32, 72, 61, 1, 67, 5, 50, 16, 25, 75, 74, 3, 44, 28, 36, 19, 69, 59, 70, 35, 68, 60, 63, 41, 40, 42],
            [27, 73, 29, 47, 26, 51, 33, 66, 45, 7, 34, 6, 52, 10, 65, 58, 13, 18, 53, 12, 56, 14, 4, 46, 20, 21],
            [55, 22, 48, 23, 17, 31, 39, 11, 71, 57, 9, 37, 64, 30, 38, 8, 24, 54, 49, 43, 2, 15, 62],
        ],
    )
    draw(
        name="Genetic algorithm: eil76, 5 salesmen",
        cities=eil76.values.copy(),
        home_city=eil76.values[0],
        tours=[
            [1, 3, 51, 12, 53, 56, 14, 4, 28, 26, 33, 6, 11, 25, 75, 5],
            [32, 8, 30, 9, 37, 64, 65, 10, 52, 57, 71, 38, 43, 2, 15, 62],
            [72, 67, 73, 70, 35, 36, 19, 69, 59, 68, 60, 63, 41, 40, 42],
            [55, 22, 48, 23, 17, 54, 24, 49, 31, 39, 16, 50],
            [21, 27, 20, 46, 47, 29, 44, 7, 18, 13, 58, 34, 45, 66, 74, 61],
        ],
    )

    draw(
        name="Genetic algorithm: rat99, 3 salesmen",
        cities=rat99.values.copy(),
        home_city=rat99.values[0],
        tours=[
            [
                1,
                10,
                12,
                21,
                20,
                23,
                33,
                32,
                31,
                40,
                39,
                47,
                56,
                74,
                73,
                64,
                55,
                57,
                60,
                61,
                53,
                62,
                70,
                69,
                68,
                67,
                66,
                65,
                54,
                46,
                38,
                37,
                27,
                18,
            ],
            [
                9,
                19,
                29,
                30,
                49,
                48,
                58,
                77,
                78,
                87,
                76,
                75,
                83,
                84,
                85,
                95,
                96,
                97,
                98,
                89,
                88,
                80,
                71,
                52,
                51,
                41,
                42,
                34,
                25,
                13,
                2,
            ],
            [
                3,
                4,
                5,
                6,
                7,
                8,
                17,
                16,
                15,
                14,
                22,
                24,
                26,
                35,
                43,
                44,
                50,
                59,
                79,
                86,
                94,
                93,
                92,
                91,
                90,
                82,
                81,
                72,
                63,
                45,
                36,
                28,
                11,
            ],
        ],
    )
    draw(
        name="Genetic algorithm: rat99, 5 salesmen",
        cities=rat99.values.copy(),
        home_city=rat99.values[0],
        tours=[
            [1, 11, 10, 28, 39, 59, 79, 78, 87, 95, 96, 88, 97, 98, 89, 80, 71, 62, 52, 41, 21],
            [3, 13, 14, 15, 24, 23, 31, 40, 32, 7, 8, 17, 26, 35, 53, 51, 38, 36, 18, 19, 9],
            [27, 45, 63, 74, 86, 94, 85, 83, 84, 75, 76, 77, 68, 67, 66, 57, 48, 47, 29],
            [37, 46, 54, 56, 55, 64, 73, 82, 92, 93, 91, 90, 81, 72, 65, 58, 49, 30, 20, 12],
            [4, 5, 6, 16, 33, 42, 50, 60, 70, 69, 61, 44, 43, 34, 25, 22, 2],
        ],
    )


def draw_particle_swarm_optimisation_algorithm(eil51, berlin52, eil76, rat99):
    draw(
        name="Particle swarm optimisation algorithm: eil51, 3 salesmen",
        cities=eil51.values.copy(),
        home_city=eil51.values[0],
        tours=[
            [5, 13, 24, 12, 40, 39, 18, 41, 43, 16, 3, 17, 46, 11, 31],
            [45, 50, 26, 47, 22, 23, 42, 6, 25, 7, 30, 27, 2, 35, 34, 19, 21],
            [1, 15, 28, 20, 33, 49, 8, 48, 9, 29, 38, 32, 44, 14, 36, 4, 37, 10],
        ],
    )
    draw(
        name="Particle swarm optimisation algorithm: eil51, 5 salesmen",
        cities=eil51.values.copy(),
        home_city=eil51.values[0],
        tours=[
            [7, 25, 30, 27, 1, 10, 37, 48, 4, 45, 11],
            [50, 13, 12, 24, 23, 42, 6, 22, 26, 31],
            [8, 9, 14, 43, 44, 32, 38, 29, 33, 49, 15],
            [20, 28, 19, 34, 35, 2, 21],
            [36, 16, 41, 18, 39, 40, 3, 46, 17, 5, 47],
        ],
    )

    draw(
        name="Particle swarm optimisation algorithm: berlin52, 3 salesmen",
        cities=berlin52.values.copy(),
        home_city=berlin52.values[0],
        tours=[
            [30, 17, 2, 16, 20, 41, 6, 1, 29, 22, 19, 49, 15, 45, 21],
            [48, 38, 36, 37, 47, 23, 14, 5, 3, 42, 32, 9, 8, 7, 40, 18, 44, 31],
            [35, 34, 33, 43, 28, 46, 25, 27, 26, 12, 13, 51, 10, 50, 11, 24, 4, 39],
        ],
    )
    draw(
        name="Particle swarm optimisation algorithm: berlin52, 5 salesmen",
        cities=berlin52.values.copy(),
        home_city=berlin52.values[0],
        tours=[
            [31, 19, 29, 1, 6, 41, 20, 16, 17, 30, 21],
            [14, 5, 3, 50, 10, 51, 13, 12, 26, 27, 45, 43],
            [23, 42, 4, 24, 25, 46, 28, 15, 49, 22],
            [48, 38, 11, 32, 9, 8, 7, 40, 18, 44, 2],
            [33, 36, 47, 37, 39, 35, 34],
        ],
    )

    draw(
        name="Particle swarm optimisation algorithm: eil76, 3 salesmen",
        cities=eil76.values.copy(),
        home_city=eil76.values[0],
        tours=[
            [5, 74, 66, 33, 51, 7, 18, 53, 12, 56, 14, 44, 28, 4, 36, 19, 69, 59, 70, 35, 68, 60, 21],
            [27, 20, 46, 47, 26, 45, 6, 34, 52, 13, 58, 10, 65, 64, 37, 9, 57, 71, 39, 11, 25, 3, 29, 73, 1, 61],
            [62, 42, 41, 63, 40, 55, 22, 48, 23, 17, 49, 24, 54, 30, 38, 8, 31, 43, 2, 15, 50, 16, 75, 67, 32, 72],
        ],
    )
    draw(
        name="Particle swarm optimisation algorithm: eil76, 5 salesmen",
        cities=eil76.values.copy(),
        home_city=eil76.values[0],
        tours=[
            [72, 32, 5, 74, 29, 44, 28, 14, 19, 36, 35, 46, 20, 73, 27, 61],
            [1, 3, 47, 4, 69, 59, 70, 68, 60, 21, 63, 41, 40, 42, 22, 55],
            [62, 48, 23, 17, 24, 38, 71, 57, 64, 65, 58, 34, 25, 66, 67],
            [51, 26, 56, 12, 53, 52, 6, 39, 8, 43, 2, 15],
            [50, 16, 11, 31, 49, 54, 30, 9, 37, 10, 13, 18, 7, 45, 33, 75],
        ],
    )

    draw(
        name="Particle swarm optimisation algorithm: rat99, 3 salesmen",
        cities=rat99.values.copy(),
        home_city=rat99.values[0],
        tours=[
            [
                16,
                26,
                43,
                44,
                53,
                52,
                51,
                50,
                49,
                59,
                60,
                61,
                62,
                71,
                80,
                79,
                70,
                69,
                68,
                76,
                67,
                58,
                57,
                48,
                38,
                40,
                41,
                42,
                31,
                30,
                28,
                19,
                9,
            ],
            [
                1,
                2,
                3,
                13,
                4,
                5,
                6,
                7,
                8,
                17,
                15,
                14,
                22,
                23,
                24,
                25,
                34,
                35,
                33,
                32,
                39,
                47,
                37,
                36,
                27,
                29,
                20,
                21,
                12,
                10,
                11,
            ],
            [
                18,
                45,
                54,
                63,
                73,
                72,
                81,
                82,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                89,
                88,
                87,
                86,
                85,
                84,
                83,
                74,
                75,
                78,
                77,
                66,
                65,
                64,
                55,
                56,
                46,
            ],
        ],
    )
    draw(
        name="Particle swarm optimisation algorithm: rat99, 5 salesmen",
        cities=rat99.values.copy(),
        home_city=rat99.values[0],
        tours=[
            [2, 12, 21, 32, 41, 43, 35, 34, 25, 24, 23, 22, 14, 15, 7, 6, 13, 3, 1],
            [18, 30, 56, 64, 74, 84, 87, 88, 78, 77, 68, 61, 50, 49, 26, 16, 17, 8, 5, 4, 11],
            [9, 10, 19, 20, 29, 27, 36, 28, 37, 48, 59, 79, 62, 53, 51, 40, 31],
            [33, 42, 44, 52, 60, 69, 70, 71, 80, 89, 98, 97, 96, 85, 92, 83, 75, 76, 65, 63],
            [73, 72, 81, 82, 90, 91, 93, 94, 95, 86, 67, 66, 58, 57, 55, 54, 45, 46, 47, 39, 38],
        ],
    )


if __name__ == "__main__":
    main()
