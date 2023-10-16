#!/usr/bin/env python
import csv
import math
import numpy as np


def main():
    output = """
\\begin{table}[!htbp]
\\caption{Results}
\\centering
\\begin{tabular}{|c c|c c c c|}
\\hline
   Alg. & Dims. & Minimum & Mean & Median & Maximum \\\\
\\hline
"""

    datasets = [
        (
            function[0],
            [
                (
                    dimension,
                    [
                        f"{function[0]}(dimensions = {dimension}) - Hybrid algorithm results: {function[1]} - run {index}.csv"
                        for index in range(0, 4)
                    ],
                )
                for dimension in [2, 30, 100]
            ],
        )
        for function in [
            ("Rastrigin", "Runtime"),
            ("Rosenbrock valley", "Runtime"),
            ("Michalewicz", "Runtime"),
            ("Griewangk", "Runtime"),
        ]
    ]

    for function, dataset in datasets:
        output += "\\multirow{3}{*}{" + function + "} "

        for dimension, function in dataset:
            data = []

            for run in function:
                last_item = "NaN"

                try:
                    with open(f"../outputs/{run}", "r", encoding="utf-8") as file:
                        reader = csv.reader(file)
                        next(reader)

                        for row in reader:
                            last_item = row[1]
                except:
                    pass

                data.append(last_item)

            data = [float(value) / 1_000_000 for value in data]
            data = list(
                filter(
                    lambda value: not math.isnan(value) and not math.isinf(value), data
                )
            )
            data.sort()
            data = np.array(data)

            if len(data) == 0:
                output += f"& {dimension}" + " & DnF" * 4 + " \\\\\n"
                continue

            minimum = np.min(data)
            mean = np.mean(data)
            median = np.median(data)
            maximum = np.max(data)

            output += f"& {dimension} & {minimum:.3} & {mean:.3} & {median:.3} & {maximum:.3} \\\\\n"
        output += "\\hline\n"

    output += """\\end{tabular}
\\end{table}
"""

    print(output)


if __name__ == "__main__":
    main()
