import os
import typing as t
import matplotlib.pyplot as plt
from functions.combinatorial.tsp.util.data import City
from functions.combinatorial.tsp.util.common import InstanceAugmenter, InstanceTransformer

exports_path = os.path.join(os.path.dirname(__file__), f"../../outputs/images")


def draw(name: str, cities: t.List[City], home_city: City, tours: t.List[t.List[int]]):
    mapper = map_tours
    mapper = InstanceAugmenter(fn=mapper, home_city=home_city)
    mapper = InstanceTransformer(fn=mapper, cities=cities)

    tours_coordinates = mapper(salesmen_routes=tours)

    plt.figure(1)
    for tour in tours_coordinates:
        xs, ys = zip(*tour)
        plt.plot(xs, ys)

    plt.title(name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(
        f"{exports_path}/{name}.png",
        format="png",
    )
    plt.close(1)


def map_tours(salesmen_routes: t.List[t.List[City]]) -> t.List[t.List[t.Tuple[float, float]]]:
    return [[(city.x, city.y) for city in route] + [(route[0].x, route[0].y)] for route in salesmen_routes]
