"""
Deep Q Network training loop for the Twixt board game.
Integrates with the existing twixt.py logic and twixtui.py visualization.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import time
import os
import pickle

from twixt import TwixtEnvironment
import twixtui
from easygraphics import delay_jfps

# ========================================
# Hyperparameters
# ========================================

BOARD_SIZE = 24
CHANNELS = 9
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000
MIN_REPLAY = 5_000
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200_000
VISUALIZE = True          # ✅ Toggle visualization here
VISUAL_FPS = 2            # frames per second for updates (2 = ~0.5s per move)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# Replay Buffer
# ========================================

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Allow terminal transitions with no valid action (use -1 as placeholder)
        if action is None:
            action = -1
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)



# ========================================
# Neural Network
# ========================================

class DQN(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, in_channels=CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 1)
        self.board_size = board_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x.view(x.size(0), -1)


# ========================================
# Twixt Environment Wrapper
# ========================================

class TwixtDQNEnv:
    """
    Wrapper connecting TwixtEnvironment to the DQN.
    Uses rotate_board() to normalize geometry for the current player.
    """
    def __init__(self, board_size=BOARD_SIZE):
        self.env = TwixtEnvironment(board_size)
        self.board_size = board_size
        self.done = False

    def reset(self):
        board, current_player, winner = self.env.reset()
        self.done = False
        # ensure player 1's perspective after reset
        if current_player == -1:
            self.env.rotate_board()
        return self._get_obs()

    def _get_obs(self):
        """
        Get the board as a (C,H,W) numpy array from the current player's perspective.
        The underlying TwixtEnvironment has already been rotated so that
        current_player is always +1 (Red orientation).
        """
        board = self.env.board.astype(np.float32)
        return np.transpose(board, (2, 0, 1))

    def step(self, action_index):
        """
        Take one move in the Twixt environment.
        Automatically handles rotation so the next player always views the board
        in their own (Red) orientation.
        """
        r = action_index // self.board_size
        c = action_index % self.board_size
        position = (r, c)
        current_player = self.env.current_player

        legal_moves = self.env.get_all_legal_moves(current_player)
        if position not in legal_moves:
            # Illegal move = small penalty, same player continues
            return self._get_obs(), -0.5, False, {'illegal': True}

        # Make the move (this may internally switch players or end the game)
        self.env.add_peg(position)
        done = self.env.winner is not None
        reward = 0.0
        info = {}

        if done:
            if self.env.winner == current_player:
                reward = 1.0
            elif self.env.winner == 0:
                reward = 0.0
            else:
                reward = -1.0

            # Store both players' terminal rewards
            info["terminal_rewards"] = {
                current_player: reward,
                -current_player: -reward
            }

        # If the game continues, rotate for the next player’s turn
        if not done and self.env.current_player == -1:
            self.env.rotate_board()

        self.done = done
        return self._get_obs(), reward, done, info

    def legal_actions_mask(self):
        """Return a boolean mask of legal moves (flattened board)"""
        mask = np.zeros(self.board_size * self.board_size, dtype=bool)
        for x, y in self.env.get_all_legal_moves(self.env.current_player):
            mask[x * self.board_size + y] = True
        return mask



# ========================================
# Helper Functions
# ========================================

def select_action(policy_net, state_tensor, legal_mask, steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        legal_idxs = np.flatnonzero(legal_mask)
        return int(np.random.choice(legal_idxs))
    else:
        with torch.no_grad():
            qvals = policy_net(state_tensor.to(DEVICE)).cpu().numpy().flatten()
            qvals[~legal_mask] = -1e9
            return int(np.argmax(qvals))


def compute_loss(batch, policy_net, target_net):
    # Filter out transitions with invalid states or actions
    valid_indices = [
        i for i, (s, a) in enumerate(zip(batch.state, batch.action))
        if s is not None and a != -1
    ]
    if len(valid_indices) == 0:
        # Skip update if batch is empty after filtering
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)

    # Slice valid transitions only
    states = [batch.state[i] for i in valid_indices]
    actions = [batch.action[i] for i in valid_indices]
    rewards = [batch.reward[i] for i in valid_indices]
    next_states = [batch.next_state[i] for i in valid_indices]
    dones = [batch.done[i] for i in valid_indices]

    # Convert to tensors
    state_batch = torch.from_numpy(np.stack(states)).to(DEVICE)
    action_batch = torch.tensor(actions, dtype=torch.long).to(DEVICE)
    reward_batch = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    done_batch = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

    # Compute Q(s, a)
    q_values = policy_net(state_batch)
    q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

    # Target computation
    with torch.no_grad():
        # Non-final mask for transitions that lead to another state
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
        target_q = torch.zeros_like(q_value)

        if non_final_mask.any():
            next_state_batch = torch.from_numpy(
                np.stack([s for s in next_states if s is not None])
            ).to(DEVICE)

            next_q_policy = policy_net(next_state_batch)
            next_actions = torch.argmax(next_q_policy, dim=1)
            next_q_target = target_net(next_state_batch)
            next_q_vals = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Bellman update
            target_q[non_final_mask] = reward_batch[non_final_mask] + GAMMA * next_q_vals

        # For terminal states, target = reward only
        target_q[~non_final_mask] = reward_batch[~non_final_mask]

    # Compute loss (only for valid transitions)
    loss = F.mse_loss(q_value, target_q)
    return loss


# ========================================
# Save/Resume
# ========================================

def save_checkpoint(policy_net, target_net, optimizer, replay, steps_done, episode, filename="checkpoint.pth"):
    """Save all relevant training state to resume later."""
    torch.save({
        'policy_state': policy_net.state_dict(),
        'target_state': target_net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'steps_done': steps_done,
        'episode': episode,
        'replay': replay.buffer  # deque of Transition tuples
    }, filename)
    print(f"Checkpoint saved at episode {episode} ({len(replay)} transitions).")


def load_checkpoint(policy_net, target_net, optimizer, replay, filename="checkpoint.pth"):
    """Load training state if checkpoint exists."""
    if not os.path.exists(filename):
        print("No checkpoint found — starting fresh.")
        return 0, 0  # episode, steps_done

    checkpoint = torch.load(filename, map_location=DEVICE, weights_only=False)
    policy_net.load_state_dict(checkpoint['policy_state'])
    target_net.load_state_dict(checkpoint['target_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    replay.buffer = checkpoint.get('replay', deque(maxlen=REPLAY_CAPACITY))
    steps_done = checkpoint.get('steps_done', 0)
    start_episode = checkpoint.get('episode', 0) + 1
    print(f"Loaded checkpoint: resuming from episode {start_episode}, step {steps_done}.")
    return start_episode, steps_done



# ========================================
# Training Loop
# ========================================

def train(num_episodes=10000):
    env = TwixtDQNEnv(BOARD_SIZE)
    policy_net = DQN(BOARD_SIZE, CHANNELS).to(DEVICE)
    target_net = DQN(BOARD_SIZE, CHANNELS).to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_CAPACITY)

    # Load checkpoint if it exists
    start_episode, steps_done = load_checkpoint(policy_net, target_net, optimizer, replay)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    if VISUALIZE:
        twixtui.initialize_graphics(env.env)

    for episode in range(start_episode, start_episode + num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        render_turn = True

        while not done:
            state_t = torch.from_numpy(obs).unsqueeze(0)
            legal_mask = env.legal_actions_mask()
            action = select_action(policy_net, state_t, legal_mask, steps_done)

            next_obs, reward, done, info = env.step(action)
            stored_next = None if done else next_obs
            replay.push(obs, action, reward, stored_next, done)
            obs = next_obs
            episode_reward += reward
            steps_done += 1

            # Render every other move
            if VISUALIZE and render_turn:
                twixtui.renderEnvironment(env.env, False)
            render_turn = not render_turn

            # Handle terminal loser transition
            if done and "terminal_rewards" in info:
                rewards = info["terminal_rewards"]
                current_player = env.env.current_player
                loser = -current_player
                loser_reward = rewards.get(loser, 0.0)

                loser_obs = env._get_obs()
                replay.push(loser_obs, None, loser_reward, None, True)

            # Training step
            if len(replay) > MIN_REPLAY:
                batch = replay.sample(BATCH_SIZE)
                loss = compute_loss(batch, policy_net, target_net)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                if steps_done % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {steps_done} | Replay size: {len(replay)}")

        # Save checkpoint every 100 episodes
        if episode % 1000 == 0 and episode > 0:
            save_checkpoint(policy_net, target_net, optimizer, replay, steps_done, episode)

    # Final save
    save_checkpoint(policy_net, target_net, optimizer, replay, steps_done, start_episode+num_episodes-1)
    print("Training complete. Model and replay buffer saved.")



if __name__ == "__main__":
    train(num_episodes=10)
