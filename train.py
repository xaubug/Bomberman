import numpy as np
import pickle
import events as e

from agent_code.my_agent.callbacks import state_to_tuple

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
GAMMA = 0.9  # Discount factor
ALPHA = 0.1  # Learning rate


def setup_training(self):
    """Called at the beginning of training."""
    self.logger.info("Setting up training...")
    self.q_table = {}


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Called after each step during training."""
    if old_game_state is None:
        return  # No learning without a previous state

    old_state = state_to_tuple(old_game_state)
    new_state = state_to_tuple(new_game_state)

    # Convert action to index
    action_idx = ACTIONS.index(self_action)

    # Update Q-value
    old_q_value = self.q_table.get(old_state, np.zeros(len(ACTIONS)))[action_idx]
    reward = reward_from_events(events)
    new_q_values = self.q_table.get(new_state, np.zeros(len(ACTIONS)))
    best_future_q = np.max(new_q_values)
    new_q_value = old_q_value + ALPHA * (reward + GAMMA * best_future_q - old_q_value)

    # Update the Q-table
    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    self.q_table[old_state][action_idx] = new_q_value


def end_of_round(self, last_game_state, last_action, events):
    """Called at the end of each round."""
    # Save the Q-table
    with open("/Users/i555661/PycharmProjects/bomberman_rl/agent_code/my_agent/q_table.pkl", "wb") as file:
        pickle.dump(self.q_table, file)


def reward_from_events(events):
    """Rewards based on the events that occurred."""
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 1
    if e.KILLED_OPPONENT in events:
        reward += 5
    if e.KILLED_SELF in events:
        reward -= 10
    return reward

# def reward_from_events(events):
#     reward = 0
#     if e.COIN_COLLECTED in events:
#         reward += 1
#     if e.KILLED_OPPONENT in events:
#         reward += 5
#     if e.KILLED_SELF in events:
#         reward -= 10
#     if e.MOVED_TOWARD_COIN in events:
#         reward += 0.1  # Reward for moving towards a coin
#     if e.ESCAPED_BOMB in events:
#         reward += 2  # Reward for successfully escaping a bomb blast
#     return reward