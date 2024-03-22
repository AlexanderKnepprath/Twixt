import twixt
import twixtui
import twixtdata

import tensorflow as tf
from keras import layers, models, saving

import numpy as np
import random

DEBUG_LEVEL = 1
VISUAL_MODE = True

# Critical note: For now, the engine always plays as player 1. 
# It may be necessary to use the rotate board function.
ENGINE_PLAYER = 1

# constants
BOARD_SIZE = 24
MIN_Q_VAL = -1000000
OPPONENT_ENGINE = saving.load_model('./qwixt_alpha_1.keras')
OPPONENT_ENGINE.summary()

# create twixt environment
env = twixt.TwixtEnvironment(BOARD_SIZE)

# set up visuals
if (VISUAL_MODE):
    twixtui.initialize_graphics(env)

def build_model(input_shape, num_actions):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same', name = "Scott"),
        layers.MaxPooling2D((2, 2), name = "Terry"),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name = "Daniel"),
        layers.MaxPooling2D((2, 2), name = "Tuomas"),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name = "Giovanni"),
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
    red_wins = 0
    blue_wins = 0
    draws = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        loop = 0

        while not done:
            print_if_debug(f"\nGame {episode+1}, Move {loop} (Red: {red_wins}, Blue: {blue_wins}, Draw: {draws})", 1)
            print_if_debug(f"Epsilon = {epsilon}")

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
            target_q_values = compute_target_q_values(next_state_q_values, 0.99)

            # Compute loss and update model
            print_if_debug("Training batch with target q-values", 3)
            print_if_debug(target_q_values, 4)
            loss = train_batch(mini_batch, target_q_values)

            print_if_debug(f"Loss: {loss}", 1)

            state = next_state

            # display board
            if (VISUAL_MODE):
                #if loop % 5 == 0:
                    twixtui.draw_heatmap(convert_q_indexes_to_positions(q_values), MIN_Q_VAL)
                    twixtui.renderEnvironment(env, True)
                #else:
                    #twixtui.renderEnvironment(env, False)

            # check for winners to update score
            if env.winner == 1:
                red_wins += 1
            elif env.winner == -1:
                blue_wins += 1
            elif env.winner == 0:
                draws += 1

                print(target_q_values)

            # increment loop counter
            loop += 1

        # Decay epsilon after each episode
        epsilon *= epsilon_decay

    model.save('./qwixt_alpha_2.keras')

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

    # get position from action index
    position = position_of_index(action)
    # attempt to add a peg at that position
    peg_added = env.add_peg(position)
    print_if_debug(f"Engine move at {position}", 2)

    reward, done = get_reward(env, position)

    # if the game is not over, get a move from the opponent
    if not done:
        opponent_response(env.get_current_state())

    # check to see if game ended on opponent's turn
    done = env.winner != None

    # get next game state
    next_state = env.get_current_state()

    return next_state, reward, done


def get_reward(environment, position):
    reward = 0
    done = False

    # if we win, then big reward
    if environment.winner == ENGINE_PLAYER:
        reward = 1000
        done = True

    # if we lose, then big penalty
    elif environment.winner == -ENGINE_PLAYER:
        reward = -1000
        done = True

    # if we draw, small penalty
    elif environment.winner == 0:
        reward = 0
        done = True

    else:
        reward += 20 * twixtdata.bridges_built(environment, ENGINE_PLAYER, position)
        
        reward += 20 * twixtdata.greatest_horizontal_distance_between_connected_pegs(environment, ENGINE_PLAYER, position)
        
        cardinal_pegs = twixtdata.num_cardinal_pegs(env, ENGINE_PLAYER, position)
        reward -= 10 * cardinal_pegs * cardinal_pegs

        #TODO: possibly compute other reward factors here

    return reward, done


def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        # Choose a random legal action
        legal_actions = env.get_all_legal_moves(ENGINE_PLAYER)
        action = index_of_position(random.choice(legal_actions))
    else:
        # Choose the action with the highest Q-value
        action = np.argmax(q_values)
    
    return action


def sample_mini_batch(replay_buffer, batch_size=16):
    # Randomly sample a mini-batch of experiences from the replay buffer
    mini_batch = replay_buffer.sample_batch(batch_size)

    boards = extract_boards_from_nth_element(mini_batch, 3)

    return mini_batch, boards


def compute_target_q_values(next_state_q_values, gamma=0.99):

    # Compute target Q-values based on the Bellman equation
    target_q_values = np.zeros(len(next_state_q_values[0]))

    for i in range(len(target_q_values)):
        immediate_reward, done = compute_immediate_reward(position_of_index(i))

        if done:
            target_q_values[i] = immediate_reward
        else:
            target_q_values[i] = immediate_reward + gamma * next_state_q_values[0][i]
    
    return target_q_values


"""
    Computes the immediate reward of a move
"""
def compute_immediate_reward(position):
    state = env.get_current_state()

    dummy_env = twixt.TwixtEnvironment(env.board_size)
    dummy_env.jump_to_state(state)
    try:
        dummy_env.add_peg(position)
        reward, done = get_reward(dummy_env, position)
    except ValueError:
        reward, done = dummy_env.winner * 1000, True

    return reward, done


"""
    Applies a mask to the list of q-values, dramatically reducing any value which corresponds to an illegal move.

    :param q_values: A list of q-values, generated by the neural network
    :param illegal_moves: A list of all the illegal moves in this position

    :return: A list of q values where all illegal moves have very low q-values
"""
def apply_action_mask(q_values, illegal_moves):
    for position in illegal_moves: # iterate through every illegal peg
        index = index_of_position(position) # get the q-value index
        q_values[index] = MIN_Q_VAL # I guess negative one million will do for now

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
    Converts target q values into np 2d array
"""
def convert_q_indexes_to_positions(q_values_1d):
    matrix = np.zeros((env.board_size, env.board_size))

    for i in range(len(matrix)):
        sub_matrix = np.zeros(env.board_size)

        for j in range(len(matrix)):
            sub_matrix[j] = q_values_1d[index_of_position((i,j))]

        matrix[i] = sub_matrix

    return matrix


"""
    The opponent makes a move!

    The opponent is now qwixt_alpha_1
"""
def opponent_response(state):

    # rotate board to make opponent engine player 1
    env.rotate_board()

    # get q values from engine
    q_values = OPPONENT_ENGINE.predict(np.expand_dims(env.board, axis=0))[0] 

    # apply mask
    action_mask_q_values = apply_action_mask(q_values, env.get_all_illegal_moves(ENGINE_PLAYER))

    # get position of best move according to opponent
    action = np.argmax(action_mask_q_values)
    x, y = position_of_index(action)

    # rotate position coordinates to reflect that opponent is blue
    flipped_position = (y, x)

    # set board back to original state
    env.rotate_board()

    # play move
    if not env.add_peg(flipped_position):
        raise ValueError("Opponent played illegal move")

  
## -- Replay Buffer -- ##

class ReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []

    def add_experience(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            index = np.random.randint(0, self.capacity)
            self.buffer[index] = (state, action, reward, next_state, done)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

## -- Print if Debug -- ##

def print_if_debug(string:str, level=4):
    if (DEBUG_LEVEL >= level):
        print(string)

## -- Hyperparams -- ##
num_episodes = 50
epsilon_decay = 0.98
max_size = 100
replay_buffer = ReplayBuffer(1024)

train_model(model, num_episodes, epsilon_decay, replay_buffer)