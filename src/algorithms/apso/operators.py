import typing as t

T = t.TypeVar("T")


class BaseTwoOptOperator(t.Generic[T]):
    def run(self, values: T) -> T:
        return values


class BasePathLinkerOperator(t.Generic[T]):
    def run(self, values: T) -> T:
        return values


class BaseSwapOperator(t.Generic[T]):
    def run(self, values: T) -> T:
        return values

