import os
import csv
import typing as t


def save_metrics(
    name: str, metrics: t.List[t.Tuple[int, float]], metric_names: t.Tuple[str, str]
):
    if not os.path.exists("./outputs"):
        os.mkdir("./outputs")

    with open(f"./outputs/{name}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(metric_names)

        for metric in metrics:
            writer.writerow(metric)


def load_metrics(name: str) -> t.List[t.Tuple[int, float]]:
    metrics = []

    with open(f"./outputs/{name}.csv", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            if len(row) == len(header):
                metric = (
                    int(row[0]),
                    float(row[1]),
                )
                metrics.append(metric)

    return metrics
