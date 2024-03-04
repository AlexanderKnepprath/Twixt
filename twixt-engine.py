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

# Example input shape (8x8 board size, 8 matrices for bridge directions)
input_shape = (8, 8, 8, 1)  # Height, width, depth, channels

# Assuming num_actions represents the number of possible actions the agent can take
num_actions = 2

# Build the model
model = build_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using mean squared error loss for Q-learning

# Print model summary
model.summary()

