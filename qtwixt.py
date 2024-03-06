import numpy as np
import tensorflow as tf
import twixt
 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv3D
from tensorflow.python.keras.optimizers import adam_v2
 
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# constants
BOARD_SIZE = 24

# create the environment
env = twixt.TwixtEnvironment(BOARD_SIZE)

# get the number of possible actions
action_space = env.get_all_legal_moves(env.current_player)
num_actions = len(action_space)
input_shape = (8, env.board_size, env.board_size, 1)  # Height, width, depth, channels

# build the agent
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

# build the model to find the optimal strategy
strategy = EpsGreedyQPolicy()
memory = SequentialMemory(limit = 10000, window_length = 1)
dqn = DQNAgent(model = model, nb_actions = num_actions,
               memory = memory, nb_steps_warmup = 10,
               target_model_update = 1e-2, policy = strategy)

# compile the model
dqn.compile(adam_v2.Adam(lr = 1e-3), metrics =['mae'])
 
# Visualizing the training 
dqn.fit(env, nb_steps = 5000, visualize = True, verbose = 2)

# Testing the learning agent
dqn.test(env, nb_episodes = 5, visualize = True)