"""
    Implements a neural network which can play Twixt
"""

__author__ = "Alexander Knepprath"

import twixt
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv3D, Flatten, Dense

BOARD_SIZE = 24

def reward_function(env: twixt.TwixtEnvironment, player: int):
    if hasWon(env, player): 
        return 1000
    elif hasLost(env, player):
        return -1000
    elif hasDrawn(env, player):
        return -100
    else:
        score = total_bridges(player)*5 + center_control_factor(player)*20              # reward building bridges and controlling center
        score -= total_bridges(player * -1)*3 - center_control_factor(player * -1)*5    # penalize allowing opponent to do the same       

def hasWon(env, player):
    return env.winner == player

def hasLost(env, player):
    return env.winner == player * -1

def hasDrawn(env, player):
    return env.win == 0

def total_bridges(board, player):
    bridge_count = 0
    for i in range (1, 5):
        for j in range (0, board.board_size):
            for k in range (0, board.board_size):
                if board.board_matrix[i, j, k] == player:
                    bridge_count += 1
    return bridge_count

# determines the total centrality of the player's pieces
# bridges are considered 2x as valuable as pegs
def center_control_factor(board, player):
    total_items = 0.0
    center_control_counter = 0.0
    for i in range (0, 9):
        for j in range (0, board.board_size):
            for k in range (0, board.board_size):
                if board.board_matrix[i, j, k] == player:
                    total_items += 1

                    # determine how far away the peg is from the center
                    x_distance_from_center = abs(board.board_size/2 - j)
                    y_distance_from_center = abs(board.board_size/2 - k)
                    # pythagoras
                    total_distance_from_center = np.sqrt(x_distance_from_center*x_distance_from_center + y_distance_from_center*y_distance_from_center)

                    # add 1/distance to center control counter
                    center_control_counter += 1/total_distance_from_center
    
    return center_control_counter / total_items


def build_model(input_shape):
    model = Sequential()

    # 3D Convolutional Layers
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))

    # Flatten layer to transition from convolutional layers to fully connected layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))  # Output layer with linear activation for Q-values

    return model

# Example input shape (board size, 8 matrices for bridge directions
input_shape = (8, BOARD_SIZE, BOARD_SIZE, 1)  # Height, width, depth, channels

# Assuming num_actions represents the number of possible actions the agent can take
num_actions = 2

# Build the model
model = build_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using mean squared error loss for Q-learning

# Print model summary
model.summary()



