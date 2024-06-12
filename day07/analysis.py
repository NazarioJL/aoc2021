from __future__ import annotations

import argparse
import operator
from statistics import mean
from statistics import median
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.optimizer_v2.adamax import Adamax
from keras.optimizer_v2.gradient_descent import SGD
from matplotlib.lines import Line2D
from tensorflow.python.client.session import Session
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.training import gradient_descent


def cost_fn_1(n: int) -> int:
    return n


def cost_fn_2(n: int) -> int:
    return (n * n + n) // 2


def total_cost(new_pos: int, pos: list[int], cost: Callable[[int], int]) -> int:
    return sum(cost(abs(p - new_pos)) for p in pos)


def main(positions: list[int]) -> int:
    max_position = max(positions)
    min_position = min(positions)

    mean_position = mean(positions)
    median_position = median(positions)

    possible_pos = range(min_position, max_position + 1)

    cost_1: dict[int, int] = {
        new_pos: total_cost(new_pos, positions, cost_fn_1) for new_pos in possible_pos
    }
    cost_2: dict[int, int] = {
        new_pos: total_cost(new_pos, positions, cost_fn_2) for new_pos in possible_pos
    }

    solution_1: tuple[int, int] = min(cost_1.items(), key=operator.itemgetter(1))
    solution_2: tuple[int, int] = min(cost_2.items(), key=operator.itemgetter(1))

    print(f"{solution_1=}, {solution_2=}")
    print(f"{mean_position=}, {median_position=}")

    x_1, y_1 = np.array(list(cost_1.keys())), np.array(list(cost_1.values()))
    x_2, y_2 = np.array(list(cost_2.keys())), np.array(list(cost_2.values()))

    fig, ax_hist = plt.subplots()
    ax_part_1 = ax_hist.twinx()
    ax_part_2 = ax_hist.twinx()

    ax_hist.set_xlabel("Position")
    ax_hist.set_ylabel("Original Position Distribution")
    ax_part_1.set_ylabel("Part 1 Cost")
    ax_part_2.set_ylabel("Part 2 Cost")

    ax_hist.hist(
        np.array(positions), alpha=0.3, color="cyan", label="Position Distribution"
    )

    (p1,) = ax_part_1.plot(x_1, y_1, color="r", label="Part 1 Cost")
    ax_part_1.plot(
        [solution_1[0]],
        [solution_1[1]],
        marker="o",
        markersize="5",
        markeredgecolor="black",
        color="r",
    )
    ax_part_1.annotate(
        str(solution_1),
        xy=solution_1,
        xytext=(-20, -10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    (p2,) = ax_part_2.plot(x_2, y_2, color="b", label="Part 2 Cost")
    ax_part_2.plot(
        [solution_2[0]],
        [solution_2[1]],
        marker="o",
        markersize="5",
        markeredgecolor="black",
        color="b",
    )
    ax_part_2.annotate(
        str(solution_2),
        xy=solution_2,
        xytext=(20, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    mean_line = plt.axvline(
        x=mean_position,
        ymin=0.0,
        ymax=1.0,
        label=f"Mean({mean_position})",
        color="green",
        ls="--",
    )
    median_line = plt.axvline(
        x=median_position,
        ymin=0.0,
        ymax=1.0,
        label=f"Median({median_position})",
        color="orange",
        ls=":",
    )

    lns = [
        Line2D([], [], c="cyan", label="Position Distribution"),
        p1,
        p2,
        median_line,
        mean_line,
    ]
    ax_hist.legend(handles=lns, loc="best")

    ax_part_2.spines["right"].set_position(("outward", 60))
    fig.tight_layout()

    # offset does not move the multiplier make it part of the label
    ax_part_1.yaxis.offsetText.set_visible(False)
    ax_part_2.yaxis.offsetText.set_visible(False)

    part1_multiplier = ax_part_1.yaxis.get_major_formatter().get_offset()
    part2_multiplier = ax_part_2.yaxis.get_major_formatter().get_offset()

    if part1_multiplier:
        new_label = f"{ax_part_1.yaxis.get_label_text()} ({part1_multiplier})"
        ax_part_1.yaxis.set_label_text(new_label)

    if part2_multiplier:
        new_label = f"{ax_part_2.yaxis.get_label_text()} ({part2_multiplier})"
        ax_part_2.yaxis.set_label_text(new_label)

    plt.show()

    return 0


def main_2(positions: list[int]) -> int:
    print(positions)
    average = mean(positions)

    x = tf.Variable(average, trainable=True)
    f = tf.add_n([tf.abs(tf.subtract(x, p)) for p in positions])

    @tf.function
    def f_x1():
        return tf.add_n([tf.abs(tf.subtract(x, p)) for p in positions])

    @tf.function
    def f_x2():
        return tf.add_n(
            [
                tf.add(tf.pow(tf.abs(tf.subtract(x, p)), 2), tf.abs(tf.subtract(x, p)))
                for p in positions
            ]
        )

    @tf.function
    def cost_part2():
        return tf.add_n([tf.abs(tf.subtract(x, p)) for p in positions])

    # x = tf.Variable(0.0, trainable=True)

    # @tf.function
    # def f_x():
    #     return tf.add_n([2 * x * x - 5 * x + 4, x * x * x])

    # for _ in range(200):
    #     print([x.numpy(), cost().numpy()])
    #     opt = gradient_descent.GradientDescentOptimizer(0.1).minimize(cost)

    opt = SGD(learning_rate=0.0003, momentum=0.0)

    f = f_x1

    var = tf.Variable(500.0, trainable=True)
    loss_1 = lambda: sum(abs(var - p) for p in positions)
    loss_2 = lambda: sum(
        ((var - p) * (var - p) + abs(var - p)) / 2.0 for p in positions
    )

    loss = loss_2
    for _ in range(25):
        step_count = opt.minimize(loss, [var])
        print(print(f"{var.numpy()=}, {loss().numpy()=},{step_count.numpy()=}"))

    ideal_n = int(round(var.numpy()))
    # var = tf.Variable(float(ideal_n))
    print(var)
    var.assign(float(ideal_n))
    var = tf.Variable(float(ideal_n), trainable=True)
    print(var)
    total = loss().numpy()
    total_2 = sum(
        ((ideal_n - p) * (ideal_n - p) + abs(ideal_n - p)) / 2 for p in positions
    )
    print(f"{ideal_n=} - {total=} - {total_2=}")

    return 0


def main_3(positions: list[int]) -> int:
    print(positions)
    average = mean(positions)

    def cost_1(n: int, ps: list[int]) -> int:
        return sum(abs(p - n) for p in ps)

    def cost_2(n: int, ps: list[int]) -> int:
        diffs = [abs(p - n) for p in ps]
        return sum((d * d + d) // 2 for d in diffs)

    def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
        vector = start
        for _ in range(n_iter):
            diff = -learn_rate * gradient(vector)
            if np.all(np.abs(diff) <= tolerance):
                break
            vector += diff
        return vector

    sol = gradient_descent(gradient=lambda n: n * 2 + 1 / 2, start=3.0, learn_rate=0.2)

    c = cost_1(sol, positions)

    print(f"{sol=}, {c=}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", default="16, 1, 2, 0, 4, 2, 7, 1, 2, 14")
    group.add_argument("-f")
    group.add_argument("-r")

    args = parser.parse_args()

    data = None

    if args.f:
        with open(args.f) as f:
            data = f.read()
    else:
        data = args.s

    data_positions = [int(n_s) for n_s in data.split(",")]  # type: ignore

    raise (SystemExit(main_2(positions=data_positions)))
# 331, 333755
# 465, 94017638
