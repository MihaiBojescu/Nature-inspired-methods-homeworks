import os
import csv
import typing as t


def save_metrics(
    name: str, metrics: t.List[t.Tuple[int, any]], metric_names: t.Tuple[str, str]
):
    if not os.path.exists("./outputs"):
        os.mkdir("./outputs")

    run = 0
    existing_metrics = os.listdir('./outputs')
    while f"{name} - run {run}.csv" in existing_metrics:
        run += 1

    with open(f"./outputs/{name} - run {run}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(metric_names)

        for metric in metrics:
            writer.writerow(metric)


def load_metrics(name: str) -> t.List[t.Tuple[int, any]]:
    metrics = []

    with open(f"./outputs/{name}.csv", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            if len(row) == len(header):
                metric = (
                    int(float(row[0])),
                    row[1],
                )
                metrics.append(metric)

    return metrics
