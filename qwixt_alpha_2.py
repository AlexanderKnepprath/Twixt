import twixt
import twixtui
import twixtdata

import tensorflow as tf
from keras import layers, models, saving

import numpy as np
import random

# Debug
DEBUG_LEVEL = 1
VISUAL_MODE = True

# File Constants
BOARD_SIZE = 24
BUFFER_CAPACITY = 100
DEFAULT_NUM_EPISODES = 100
DEFAULT_EPSILON_DECAY = 0.98
DEFAULT_GAMMA = 1

# QwixtAlpha2 Constants
ENGINE_PLAYER = 1 # <- DO NOT ALTER
BATCH_SAMPLE_SIZE = 32

class QwixtAlpha2:

    def __init__(self, board_size:int, buffer_capacity:int, opponent_engine, minimum_q_val=-1000000):
        self.environment = twixt.TwixtEnvironment(board_size=board_size)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.model = self.build_model((board_size, board_size, 9), board_size*board_size)
        self.opponent_engine = opponent_engine
        self.min_q_val = minimum_q_val

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error')


    """
        Creates the NN model
    """
    def build_model(self, input_shape, num_actions):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_actions, activation='linear')  # Output layer for Q-values
        ])
        return model
    
    """
        Trains the NN model
    """
    def train_model(self, num_episodes, epsilon_decay=0.98, gamma=1):
        # shorthand
        env = self.environment
        model = self.model

        # initialize epsilon
        epsilon = 1

        # initialize data
        model_wins = 0
        opponent_wins = 0
        draws = 0

        # For each episode
        for episode in range(num_episodes):

            # Initialize the starting state
            state = env.reset()
            done = False
            loop = 0

            # For each timestep
            while not done:
                # Header
                print_if_debug(f"\nGame {episode+1}, Move {loop} (Model: {model_wins}, Opponent: {opponent_wins}, Draw: {draws})", 1)

                # Compute q-values for all actions
                q_values = self.predict_q_values()

                # Apply action mask to q-values
                illegal_actions = env.get_all_illegal_moves(ENGINE_PLAYER)
                masked_q_values = self.apply_action_mask(q_values, illegal_actions)
                
                # Select action via exploration or exploitation
                action = self.epsilon_greedy_policy(masked_q_values, epsilon)

                # Execute selected action in an emulator, observe reward and next state
                next_state, reward, done = self.step(action, state)

                # Store experience in replay memory
                self.replay_buffer.add_experience(state, q_values, action, reward, next_state, done)

                # Sample random batch from replay memory
                random_batch = self.replay_buffer.sample_batch(BATCH_SAMPLE_SIZE)

                # Preprocess states from batch
                states = extract_boards_from_nth_elements(random_batch, 0) # q-value predicted for each state-action pair
                rewards = extract_nth_elements(random_batch, 3) # reward recieved for each state-action pair
                next_boards = extract_boards_from_nth_elements(random_batch, 4) # board of next state after action taken
                dones = extract_nth_elements(random_batch, 5) # whether the state was terminal after action

                # Assuming you have already defined your state and selected action
                for i in range(len(random_batch)):

                    # calculate target Q-value
                    target_q_value = rewards[i] 
                    if not dones[i]:
                        target_q_value += gamma * model.predict(np.expand_dims(next_boards[i], axis=0))[0][action]

                    # initialize target q values array and set the selected action to the q value we want
                    target_q_values = np.zeros(env.board_size * env.board_size)
                    target_q_values[action] = target_q_value

                    # Create sample weights
                    sample_weights = np.zeros_like(q_values)
                    sample_weights[action] = 1.0  # Assign non-zero weight to the selected action

                    # Train the model using train_on_batch
                    loss = model.train_on_batch(np.expand_dims(states[i], axis=0), np.array(target_q_values), sample_weight=sample_weights)
                    
                    
            # decay epsilon
            epsilon *= epsilon_decay


    ### -------------------------- ###
    ### - Training Sub-functions - ###
    ### -------------------------- ###

    
    """
        Predicts q_values and returns as either a 1d or 2d np array

        :param as2d: Function returns 2d array iff this param is set to "True"
    """
    def predict_q_values(self, as2d:bool=False):
        q_values = self.model.predict(np.expand_dims(self.environment.board, axis=0))[0]

        if as2d:
            q_values_2d = np.zeros((self.environment.board_size, self.environment.board_size))

            for i in range(len(self.environment.board_size)):
                for j in range(len(self.environment.board_size)):
                    q_values_2d[i][j] = q_values[index_of_position((i, j), self.environment)]
            
            return q_values_2d
        
        else:
            return q_values
        

    """
        Applies a mask to the list of q-values, dramatically reducing any value which corresponds to an illegal move.

        :param q_values: A list of q-values, generated by the neural network
        :param illegal_moves: A list of all the illegal moves in this position

        :return: A list of q values where all illegal moves have very low q-values
    """
    def apply_action_mask(self, q_values, illegal_moves):
        for position in illegal_moves: # iterate through every illegal peg
            index = index_of_position(position, self.environment) # get the q-value index
            q_values[index] = self.min_q_val

        return q_values
    

    """
        Chooses an action based on exploration /or/ exploitation, 
        depending on a random number's relation to epsilon

        :returns: index of the action
    """
    def epsilon_greedy_policy(self, q_values, epsilon):
        if np.random.rand() < epsilon:
            # Choose a random legal action
            legal_actions = self.environment.get_all_legal_moves(ENGINE_PLAYER)
            action = index_of_position(random.choice(legal_actions), self.environment)
        else:
            # Choose the action with the highest Q-value
            action = np.argmax(q_values)
        
        return action


    """
        Takes a chosen action from a given state

        :return: A tuple
        :return[0]: The next state after opponent action,
        :return[1]: The reward for this action
        :return[2]: Whether the game haas ended after this action
    """
    def step(self, action, state):
        env = self.environment

        # first, load the state
        env.jump_to_state(state)

        # verify that it's actually the engine's turn to play
        if (env.current_player != ENGINE_PLAYER):
            raise Exception("The engine is trying to make a move, but it is not the engine's turn")

        # get position from action index
        position = position_of_index(action, env)
        # attempt to add a peg at that position
        peg_added = env.add_peg(position)
        print_if_debug(f"Engine move at {position}", 2)

        reward, done = self.compute_reward(env.get_current_state())

        # if the game is not over, get a move from the opponent
        if not done:
            self.opponent_response(env.get_current_state())

        # check to see if game ended on opponent's turn
        done = env.winner != None

        # get next game state
        next_state = env.get_current_state()

        return next_state, reward, done
    

    """
        Computes the reward based on current environment state
    """
    def compute_reward(self, state):
        env = self.environment
        env.jump_to_state(state)

        if env.winner == ENGINE_PLAYER:
            return 1000, True
        elif env.winner == -ENGINE_PLAYER:
            return -1000, True
        elif env.winner == 0:
            return -100, True
        else:
            return 0, False
        
    
    """
        Gets response from opponent
    """
    def opponent_response(self, state):
        # define environment and opponent
        env = self.environment
        opp = self.opponent_engine
        env.jump_to_state(state)

        # rotate board for opponent to make a move
        env.rotate_board()

        # get y and x coordinates (translated because board is rotated)
        y, x = opp.position_of_best_move(env.get_current_state())        

        env.rotate_board()

        env.add_peg((x,y))
        

    ### ---------------------- ###
    ### - Gameplay functions - ###
    ### ---------------------- ###

    """
        Returns the best move (according to the engine) in the position for player ENGINE_PLAYER.

        !!! IMPORTANT !!! - If engine is playing as player -1 (player 2), it is required to
        first rotate the board before calling this function!
    """
    def position_of_best_move(self, state):
        env = self.environment
        env.jump_to_state(state)

        q_values = self.predict_q_values()
        masked_q_values = self.apply_action_mask(q_values, env.get_all_illegal_moves(ENGINE_PLAYER))

        return position_of_index(np.argmax(masked_q_values), env)


### --------------------------- ###
### - Useful helper functions - ###
### --------------------------- ###

# get action space index of 2d location
def index_of_position(position, environment):
    return position[0] * environment.board_size + position[1]

# get the 2d peg location from action space index
def position_of_index(index, environment):
    return (index // environment.board_size, index % environment.board_size)


def extract_boards_from_nth_elements(list_of_tuples, n):
    states = list()
    for i in list_of_tuples:
        states.append(i[n])
    
    boards = list()
    for i in states:
        boards.append(i[0])

    boards = np.array(boards)

    return boards


def extract_nth_elements(list_of_tuples, n):
    elements = list()
    for i in list_of_tuples:
        elements.append(i[n])

    return elements


class ReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []

    def add_experience(self, state, q_values, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append([state, q_values, action, reward, next_state, done])
        else:
            index = np.random.randint(0, self.capacity)
            self.buffer[index] = [state, q_values, action, reward, next_state, done]

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    

class RandomOpponent:

    def __init__(self, board_size):
        self.env = twixt.TwixtEnvironment(board_size)

    """
        Returns a random legal move for player ENGINE_PLAYER (player 1)

        !!! IMPORTANT !!! - If engine is playing as player -1 (player 2), it is required to
        first rotate the board before calling this function!
    """
    def position_of_best_move(self, state):
        self.env.jump_to_state(state)

        legal_moves = self.env.get_all_legal_moves(ENGINE_PLAYER)

        return legal_moves[np.random.randint(0, len(legal_moves))]
    


def print_if_debug(string:str, level):
    if (DEBUG_LEVEL >= level):
        print(string)


# Initialize replay memory capacity
# Initialize network with random weights
engine = QwixtAlpha2(BOARD_SIZE, BUFFER_CAPACITY, RandomOpponent(BOARD_SIZE))

# Training loop
engine.train_model(DEFAULT_NUM_EPISODES, DEFAULT_EPSILON_DECAY, DEFAULT_GAMMA)
