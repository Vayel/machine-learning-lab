import random as rd
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

ROW_NUM = 2
COL_NUM = 3


def mse(data, f):
    n = data.size
    return 1/n * sum((f(x) - y)**2 for x, y in data)


def mse_gradient(data, f):
    n = data.size
    d_a = sum(2/n * x * (f(x) - y) for x, y in data)
    d_b = sum(2/n * (f(x) - y) for x, y in data)

    return np.array([d_a, d_b])


def produce_data(f, n, x_min, x_max):
    for _ in range(n):
        x = rd.uniform(x_min, x_max)
        yield x, f(x)


def read_data(data):
    return data[:, 0], data[:, 1]


def linear(a, b, x):
    return a * x + b


class LinearRegression:
    def __init__(self, data):
        self.data = data
        self.model = linear
        self.cost = None
        self.gradient = None
        self.params = np.array([rd.random(), rd.random()])

    @property
    def f(self):
        return partial(linear, *self.params)

    def plot(self, t=None, error=None, learning_rate=None, params=None):
        x, y = read_data(self.data)

        plt.ion()  # Non-blocking plotting

        # Plot data with model
        plt.subplot(ROW_NUM, COL_NUM, 1).cla()

        col_mins, col_maxs = np.amin(self.data, 0), np.amax(self.data, 0)
        plt.xlim(col_mins[0], col_maxs[0])
        plt.ylim(col_mins[1], col_maxs[1])

        plt.plot(x, y, 'ro')
        plt.pause(0.001)  # Let the time to plot data before giving hand back
        plt.plot(x, np.vectorize(self.f)(x), 'b')
        plt.pause(0.001)

        if t is None:
            return

        if error is not None:
            plt.subplot(ROW_NUM, COL_NUM, 2).set_ylim(0, 1)
            plt.scatter(t, error)
            plt.pause(0.0001)

        if learning_rate is not None:
            plt.subplot(ROW_NUM, COL_NUM, 3).set_ylim(0, 0.001)
            plt.scatter(t, learning_rate)
            plt.pause(0.0001)

        if params is not None:
            for i in range(params.size):
                plt.subplot(ROW_NUM, COL_NUM, 4 + i)
                plt.scatter(t, params[i])
                plt.pause(0.0001)

    def train_step(self, gradient, learning_rate):
        gd = gradient(self.data, partial(linear, *self.params))
        self.params -= learning_rate * gd  # Element-wise operations with numpy arrays

    def train(self, cost, gradient, max_error, min_delta_error, max_iterations,
              learning_rate=0.001, learning_rate_decay=0.9):
        t = 0
        error = cost(self.data, self.f)

        while t < max_iterations and error > max_error:
            self.plot(t, error, learning_rate, self.params)
            t += 1

            self.train_step(gradient, learning_rate)

            old_error, error = error, cost(self.data, self.f)
            if abs(old_error - error) < min_delta_error:
                learning_rate *= learning_rate_decay
