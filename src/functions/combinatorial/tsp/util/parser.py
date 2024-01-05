from dataclasses import dataclass
import os
import re
import typing as t
from functions.combinatorial.tsp.util.common import City

path = os.path.join(os.path.dirname(__file__), "../../../../../")


@dataclass
class ParserResult:
    name: str
    description: str
    dimensions: int
    coordinates: t.List[City]


class Parser:
    __line_index: int

    def __init__(self) -> None:
        self.__line_index = 0

    def parse(self, file_path: str) -> ParserResult:
        self.__line_index = 0
        previous_line_index = None
        result = ParserResult(name="", description="", dimensions=0, coordinates=[])
        lines = []

        with open(f"{path}/{file_path}", encoding="utf-8") as file:
            lines = file.readlines()

        while self.__line_index < len(lines):
            if previous_line_index == self.__line_index:
                raise RuntimeError(
                    f"Parsing did not advance. Stuck at line {self.__line_index}: {lines[self.__line_index]}"
                )

            previous_line_index = self.__line_index

            if re.match(r"^NAME", lines[self.__line_index]):
                name, increment = self.__parse_name(lines, self.__line_index)
                result.name = name
                self.__line_index += increment
                continue

            if re.match(r"^COMMENT", lines[self.__line_index]):
                description, increment = self.__parse_description(lines, self.__line_index)
                result.description = description
                self.__line_index += increment
                continue

            if re.match(r"^DIMENSION", lines[self.__line_index]):
                dimensions, increment = self.__parse_dimensions(lines, self.__line_index)
                result.dimensions = dimensions
                self.__line_index += increment
                continue

            if re.match(r"^NODE_COORD_SECTION", lines[self.__line_index]):
                coordinates, increment = self.__parse_coordinates(lines, self.__line_index)
                result.coordinates = coordinates
                self.__line_index += increment
                continue

            self.__line_index += 1  # for other types of lines

        return result

    def __parse_name(self, lines: t.List[str], index: int) -> t.Tuple[str, int]:
        name = lines[index].split(":")[1].strip()
        increment = 1
        return name, increment

    def __parse_description(self, lines: t.List[str], index: int) -> t.Tuple[str, int]:
        description = lines[index].split(":")[1].strip()
        increment = 1
        return description, increment

    def __parse_dimensions(self, lines: t.List[str], index: int) -> t.Tuple[int, int]:
        dimensions = int(lines[index].split(":")[1].strip())
        increment = 1
        return dimensions, increment

    def __parse_coordinates(self, lines: t.List[str], index: int) -> t.Tuple[int, int]:
        coordinates = []
        increment = 1
        number_regex = re.compile(r"\d+[.,]?\d*")

        for i in range(index + 1, len(lines)):
            if "EOF" in lines[i]:
                increment += 1
                break

            identifier, x, y = number_regex.findall(lines[i])
            coordinates.append(City(identifier=int(identifier), x=float(x), y=float(y)))
            increment += 1

        return coordinates, increment
