# http://outlace.com/Reinforcement-Learning-Part-3/

import random

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

PLAYER = np.array([0, 0, 0, 1])
WALL = np.array([0, 0, 1, 0])
PIT = np.array([0, 1, 0, 0])
GOAL = np.array([1, 0, 0, 0])

one_pos = lambda x: np.where(x == 1)[0][0]
hash_ = one_pos

NOTHING_SYMBOL = '.'
SYMBOLS = {
    hash_(PLAYER): 'x',
    hash_(WALL): '#',
    hash_(PIT): 'o',
    hash_(GOAL): '*',
}

GRID_WIDTH = GRID_HEIGHT = 4

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS_NUMBER = 4

MODEL_INPUT_LEN = GRID_HEIGHT * GRID_WIDTH * len(PLAYER)
DEFAULT_REWARD = -1
MODEL_WEIGHTS_FILE = 'nn_weights'


def find_loc(state, obj):
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if np.logical_and(state[i, j], obj).any():
                return i, j


def find_locs(state):
    return {
        hash_(PLAYER): find_loc(state, PLAYER),
        hash_(WALL): find_loc(state, WALL),
        hash_(PIT): find_loc(state, PIT),
        hash_(GOAL): find_loc(state, GOAL),
    }


def create_state(**kwargs):
    objs = {
        'player': PLAYER,
        'goal': GOAL,
        'wall': WALL,
        'pit': PIT,
    }
    state = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(PLAYER)))

    for key, obj in objs.items():
        if kwargs.get(key):
            # logical_or allows to superimpose elements
            state[kwargs[key]] = np.logical_or(state[kwargs[key]], obj)

    return state


def init_grid():
    state = create_state(
        player=(0, 1),
        wall=(2, 2),
        pit=(1, 1),
        goal=(3, 3),
    )

    return state


def has_superimposed(state):
    return (np.sum(state, axis=2) > 1).any()


def init_grid_random_player():
    state = create_state(
        wall=(2, 2),
        pit=(1, 1),
        goal=(3, 3),
        player=(np.random.randint(0, GRID_HEIGHT), np.random.randint(0, GRID_WIDTH)),
    )

    if has_superimposed(state):
        return init_grid_random_player()

    return state


def draw(state):
    locs = find_locs(state)

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            for obj, loc in locs.items():
                if loc == (i, j):
                    print(SYMBOLS[obj], end='')
                    break
            else:
                print(NOTHING_SYMBOL, end='')
        print('')


def is_loc_valid(loc, state):
    loc = np.array(loc)
    return (
        (loc != find_loc(state, WALL)).any() and
        (loc < (GRID_HEIGHT, GRID_WIDTH)).all() and
        (loc >= (0, 0)).all()
    )


def move(state, action):
    player_loc = find_loc(state, PLAYER)
    state = create_state(
        goal=find_loc(state, GOAL),
        wall=find_loc(state, WALL),
        pit=find_loc(state, PIT),
    )

    if action == UP:
        new_loc = (player_loc[0] - 1, player_loc[1])
    elif action == DOWN:
        new_loc = (player_loc[0] + 1, player_loc[1])
    elif action == LEFT:
        new_loc = (player_loc[0], player_loc[1] - 1)
    elif action == RIGHT:
        new_loc = (player_loc[0], player_loc[1] + 1)
    else:
        raise ValueError('Invalid action: {0}'.format(action))

    if not is_loc_valid(new_loc, state):
        new_loc = player_loc

    state[new_loc] = np.logical_or(state[new_loc], PLAYER)

    return state


def measure_reward(state):
    player_loc = find_loc(state, PLAYER)

    if player_loc == find_loc(state, PIT):
        return -10
    elif player_loc == find_loc(state, GOAL):
        return 10
    else:
        return DEFAULT_REWARD


def train(model, initial_state, epochs, gamma, initial_epsilon, min_epsilon=0.1):
    epsilon = initial_epsilon

    for _ in range(epochs):
        state = initial_state

        while True:
            #We are in state S
            #Let's run our Q function on S to get Q values for all possible actions
            qval = model.predict(state.reshape(1, MODEL_INPUT_LEN), batch_size=1)
            if random.random() < epsilon: # Choose random action
                action = np.random.randint(0, ACTIONS_NUMBER)
            else: # Choose best action from Q(s,a) values
                action = (np.argmax(qval))

            new_state = move(state, action)
            reward = measure_reward(new_state)

            # Get max_Q(S',a)
            new_Q = model.predict(new_state.reshape(1, MODEL_INPUT_LEN), batch_size=1)
            max_Q = np.max(new_Q)
            y = np.zeros((1, ACTIONS_NUMBER))
            y[:] = qval[:]

            if reward == DEFAULT_REWARD: # Non-terminal state
                update = (reward + (gamma * max_Q))
            else: # Terminal state
                update = reward

            y[0][action] = update # Target output
            model.fit(state.reshape(1, MODEL_INPUT_LEN), y, batch_size=1, epochs=1, verbose=0)
            state = new_state

            if reward != DEFAULT_REWARD:
                break

        if epsilon > min_epsilon:
            epsilon -= 1 / epochs


def run(model, initial_state, max_iterations):
    i = 0
    state = initial_state
    draw(state)

    while True:
        qval = model.predict(state.reshape(1, MODEL_INPUT_LEN), batch_size=1)
        action = np.argmax(qval)
        state = move(state, action)

        print('')
        draw(state)

        if measure_reward(state) != DEFAULT_REWARD:
            break

        i += 1
        if i > max_iterations:
            print('Game lost; too many moves.')
            break


if __name__ == '__main__':
    state = init_grid_random_player()

    model = Sequential()
    model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(MODEL_INPUT_LEN,)))
    model.add(Activation('relu'))

    model.add(Dense(150, kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(4, kernel_initializer='lecun_uniform'))
    model.add(Activation('linear')) # Linear output so we can have range of real-valued outputs

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    try:
        model.load_weights(MODEL_WEIGHTS_FILE, by_name=False)
    except OSError:
        train(model, init_grid(), 1000, 0.9, 1)
        model.save_weights(MODEL_WEIGHTS_FILE)

    run(model, state, 10)
