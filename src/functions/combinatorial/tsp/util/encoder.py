import typing as t

E = t.List[int]
D = t.List[t.List[int]]


class Encoder:
    __segments: t.List[int]

    def __init__(self, segments: t.List[int]):
        self.__segments = segments

    def encode(self, value: D) -> E:
        return [city for tour in value for city in tour]

    def decode(self, value: E) -> D:
        result: E = [[0 for _ in range(segment)] for segment in self.__segments]
        index = 0

        for tour in result:
            for i, _ in enumerate(tour):
                tour[i] = value[index]
                index += 1

        return result
