import typing as t


def quicksort(
    data: t.List[any], comparator: t.Callable[[any, any], bool]
) -> t.List[any]:
    if len(data) <= 1:
        return data
    else:
        pivot = data[0]
        left = [x for x in data[1:] if comparator(x, pivot)]
        right = [x for x in data[1:] if not comparator(x, pivot)]
        return quicksort(left, comparator) + [pivot] + quicksort(right, comparator)
