from dataclasses import dataclass


@dataclass
class City:
    identifier: int
    x: float
    y: float

    def to_index(self):
        return self.identifier - 1
