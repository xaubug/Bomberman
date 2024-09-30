import numpy as np
import pickle


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']


def setup(self):
    """Called once before the game starts to initialize the agent."""
    self.logger.info("Setting up my Q-learning agent")
    self.q_table = {}

    # Load Q-table if available
    try:
        with open("/Users/i555661/PycharmProjects/bomberman_rl/agent_code/my_agent/q_table.pkl", "rb") as file:
            self.q_table = pickle.load(file)
    except FileNotFoundError:
        self.logger.info("No pre-trained Q-table found. Starting fresh.")


def act(self, game_state: dict):
    """Decides which action to take."""
    # Convert game_state to a hashable representation
    state = state_to_tuple(game_state)

    # Exploration vs Exploitation (epsilon-greedy)
    epsilon = 0.1
    #decaying epsilon
    #epsilon = max(0.1, epsilon * 0.995)  # Gradually reduce exploration

    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)  # Explore

    # Exploit: Choose the best action based on the Q-table
    q_values = self.q_table.get(state, np.zeros(len(ACTIONS)))
    return ACTIONS[np.argmax(q_values)]


def state_to_tuple(game_state):
    """Convert the game state dictionary to a hashable tuple."""
    field = tuple(map(tuple, game_state['field']))
    coins = tuple(game_state['coins'])
    position = game_state['self'][-1]
    bombs = tuple((pos, timer) for pos, timer in game_state['bombs'])
    return (field, coins, position, bombs)