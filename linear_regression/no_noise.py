"""Try to fit a perfect line (ax + b)."""

from functools import partial

import numpy as np

from common import LinearRegression, linear, produce_data, mse, mse_gradient


if __name__ == '__main__':
    MAX_ERROR = 0.001
    MIN_DELTA_ERROR = 0.001
    MAX_ITERATIONS = 10
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.9
    N = 10
    X_MIN = 0
    X_MAX = 50

    f = partial(linear, 1, 1)
    data = np.array(list(produce_data(f, N, X_MIN, X_MAX)))
    model = LinearRegression(data)
    model.train(mse, mse_gradient, MAX_ERROR, MIN_DELTA_ERROR, MAX_ITERATIONS,
                LEARNING_RATE, LEARNING_RATE_DECAY)

    input('Waiting for user input to close...')
