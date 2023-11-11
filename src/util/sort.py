import typing as t


def quicksort(data: t.List[any], comparator: t.Callable[[any, any], bool]):
    stack = [(0, len(data) - 1)]

    while stack:
        low, high = stack.pop()

        if low < high:
            pivot_index = partition(data, low, high, comparator)
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))

    return data


def partition(
    data: t.List[any], low: int, high: int, comparator: t.Callable[[any, any], bool]
):
    pivot = data[high]
    i = low - 1

    for j in range(low, high):
        if comparator(data[j], pivot):
            i += 1
            data[i], data[j] = data[j], data[i]

    data[i + 1], data[high] = data[high], data[i + 1]

    return i + 1
