"""Try to fit a perfect line (ax + b)."""

from functools import partial

import numpy as np
from sklearn import linear_model

from common import LinearRegression, linear, read_data, produce_data, mse, mse_gradient


if __name__ == '__main__':
    # Manual training
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
    print("Manual params: " + str(model.params))

    # Sk learn training
    reg = linear_model.LinearRegression()
    x, y = read_data(data)
    reg.fit(x.reshape(-1, 1), y)
    print("Sklearn params: " + str(reg.coef_) + " " + str(reg.intercept_))

    input('Waiting for user input to close...')
