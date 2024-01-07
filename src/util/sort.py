import typing as t

Data = t.TypeVar("Data")


def quicksort(data: t.List[Data], comparator: t.Callable[[Data, Data], bool]):
    stack = [(0, len(data) - 1)]

    while stack:
        low, high = stack.pop()

        if low < high:
            pivot_index = partition(data, low, high, comparator)
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))

    return data


def partition(data: t.List[Data], low: int, high: int, comparator: t.Callable[[Data, Data], bool]):
    pivot = data[high]
    i = low - 1

    for j in range(low, high):
        if comparator(data[j], pivot):
            i += 1
            data[i], data[j] = data[j], data[i]

    data[i + 1], data[high] = data[high], data[i + 1]

    return i + 1


def minimise(a: Data, b: Data) -> bool:
    return a < b


def maximise(a: Data, b: Data) -> bool:
    return a > b
