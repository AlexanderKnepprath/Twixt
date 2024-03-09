import twixt

import tensorflow as tf
from keras import layers, models

import numpy as np
import random

DEBUG_MODE = 4

# Critical note: For now, the engine always plays as player 1. 
# It may be necessary to use the rotate board function.
ENGINE_PLAYER = 1

# constants
BOARD_SIZE = 24

# create twixt environment
env = twixt.TwixtEnvironment(24)

def build_model(input_shape, num_actions):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name = "Scott"),
        layers.MaxPooling2D((2, 2), name = "Terry"),
        layers.Conv2D(64, (3, 3), activation='relu', name = "Daniel"),
        layers.MaxPooling2D((2, 2), name = "Tuomas"),
        layers.Conv2D(64, (3, 3), activation='relu', name = "Giovanni"),
        layers.Flatten(name = "Stanley"),
        layers.Dense(64, activation='relu', name = "Rein"),
        layers.Dense(num_actions, activation='linear', name = "Evgeny")  # Output layer for Q-values
    ])
    return model

# Define input shape based on state representation (nxnx9)
input_shape = (env.board_size, env.board_size, 9)

## -- ACTION SPACE FUNCTIONS -- ##

# Define the number of legal peg placements as the number of actions
num_actions = env.board_size * env.board_size

# get action space index of 2d location
def index_of_position(position):
    return position[0] * env.board_size + position[1]

# get the 2d peg location from action space index
def position_of_index(index):
    return (index // env.board_size, index % env.board_size)

## -- MODEL DESIGN -- ##

# Build the model
model = build_model(input_shape, num_actions)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print model summary
model.summary()

## -- TRAINING LOOP FUNCTIONS -- ##

# Define your training loop
def train_model(model, num_episodes, epsilon_decay, replay_buffer):
    epsilon = 1.0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        loop = 0

        while not done:
            print_if_debug(f"\nEpisode {episode}, Loop {loop}", 1)

            # Compute Q-values for all possible moves
            print_if_debug("Computing q-values", 3)
            q_values = model.predict(np.expand_dims(env.board, axis=0))[0]

            # Apply mask to Q-values to set illegal moves to a very low number
            print_if_debug("Getting illegal actions", 3)
            illegal_actions = env.get_all_illegal_moves(ENGINE_PLAYER)
            print_if_debug("Setting illegal actions' q-values to very low", 3)
            masked_q_values = apply_action_mask(q_values, illegal_actions)

            # Choose action using epsilon-greedy policy
            print_if_debug("Choosing action", 3)
            action = epsilon_greedy_policy(masked_q_values, epsilon)

            # take this action
            print_if_debug("Taking a step", 3)
            next_state, reward, done = step(action, state)
            print_if_debug("Adding experience to replay buffer", 3)
            replay_buffer.add_experience(state, action, reward, next_state, done)

            # Sample mini-batch from replay buffer
            print_if_debug("Sampling mini-batch from replay buffer", 3)
            mini_batch, batch_next_states = sample_mini_batch(replay_buffer)

            # Compute target Q-values using Bellman equation
            print_if_debug("Getting next state q-values:", 3)
            next_state_q_values = model.predict(batch_next_states)
            print_if_debug("Computing target q-values with bellman equation", 3)
            target_q_values = compute_target_q_values(mini_batch, next_state_q_values)

            # Compute loss and update model
            print_if_debug("Training batch with target q-values", 3)
            print_if_debug(target_q_values, 4)
            loss = train_batch(mini_batch, target_q_values)

            print_if_debug(f"Loss: {loss}", 1)

            state = next_state

            loop += 1

        # Decay epsilon after each episode
        epsilon *= epsilon_decay

# Helper functions
"""
    Takes a single action in the training loop

    :param action: an index 0 < x < n^2 where n is the board size

    :return: A tuple:
    :return[0]: The new state of the board
    :return[1]: The reward, an integer
    :return[2]: A boolean which is true if the round has ended
"""
def step(action, state):

    # first, load the state
    env.jump_to_state(state)

    # verify that it's actually the engine's turn to play
    if (env.current_player != ENGINE_PLAYER):
        raise Exception("The engine is trying to make a move, but it is not the engine's turn")

    # define returns
    next_state = None
    reward = 0
    done = False

    # get position from action index
    position = position_of_index(action)
    # attempt to add a peg at that position
    peg_added = env.add_peg(position)

    # theoretically this should be impossible, but if the engine tries an illegal move, we punish.
    if not peg_added:
        reward = -100

    # if a move was played, now we get a move from the opponent
    if peg_added:
        opponent_response(env.get_current_state())

    # get next game state
    next_state = env.get_current_state()
    
    # if we win, then big reward
    if env.winner == ENGINE_PLAYER:
        reward = 1000
        done = True

    # if we lose, then big penalty
    elif env.winner == -ENGINE_PLAYER:
        reward = -1000
        done = True

    # if we draw, small penalty
    elif env.winner == 0:
        reward = -100
        done = True

    #TODO: possibly compute other reward factors here

    return next_state, reward, done


def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        # Choose a random action
        action = np.random.randint(len(q_values))
    else:
        # Choose the action with the highest Q-value
        action = np.argmax(q_values)
    
    return action


def sample_mini_batch(replay_buffer, batch_size=16):
    # Randomly sample a mini-batch of experiences from the replay buffer
    mini_batch = replay_buffer.sample_batch(batch_size)

    boards = extract_boards_from_nth_element(mini_batch, 3)

    return mini_batch, boards


def compute_target_q_values(mini_batch, next_state_q_values, gamma=0.99):
    # Extract components from the mini-batch
    rewards = list()
    dones = list()

    for i in mini_batch:
        rewards.append(i[2])

    # Compute target Q-values based on the Bellman equation
    target_q_values = rewards + gamma * np.max((next_state_q_values), axis=1)

    # Update Q-values for actions that led to terminal states
    for i, done in enumerate(dones):
        if done:
            target_q_values[i] = rewards[i]
    
    return target_q_values


"""
    Applies a mask to the list of q-values, dramatically reducing any value which corresponds to an illegal move.

    :param q_values: A list of q-values, generated by the neural network
    :param illegal_moves: A list of all the illegal moves in this position

    :return: A list of q values where all illegal moves have very low q-values
"""
def apply_action_mask(q_values, illegal_moves):
    for position in illegal_moves: # iterate through every illegal peg
        index = index_of_position(position) # get the q-value index
        q_values[index] = -1000000 # I guess negative one million will do for now

    return q_values


def train_batch(mini_batch, target_q_values):

    boards = extract_boards_from_nth_element(mini_batch, 0)
    
    loss = model.train_on_batch(boards, target_q_values)

    return loss


"""
    Takes a list of tuples where the nth element in each tuple is a list of states,
    retrieves the boards from those states,
    and returns them as a numpy array

    :param tuple: The tuple to take from
    :param n: the position of the list of states to extract boards from
"""
def extract_boards_from_nth_element(list_of_tuples, n):
    states = list()
    for i in list_of_tuples:
        states.append(i[n])
    
    boards = list()
    for i in states:
        boards.append(i[0])

    boards = np.array(boards)

    return boards


"""
    The opponent makes a move!

    For now, the opponent is purely random!
"""
def opponent_response(state):
    legal_moves = env.get_all_legal_moves(-ENGINE_PLAYER)
    choice_index = np.random.randint(0, len(legal_moves))
    if not env.add_peg(legal_moves[choice_index]):
        raise Exception("Opponent played illegal move!")

  
## -- Replay Buffer -- ##

class ReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add_experience(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, min(batch_size, self.position))

## -- Print if Debug -- ##

def print_if_debug(string:str, level=4):
    if (DEBUG_MODE >= level):
        print(string)

## -- Hyperparams -- ##
num_episodes = 1000
epsilon_decay = 0.98
max_size = 100
replay_buffer = ReplayBuffer(256)

train_model(model, num_episodes, epsilon_decay, replay_buffer)