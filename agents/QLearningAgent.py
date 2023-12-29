from copy import copy

import numpy as np


class QLearningAgent:
    def __init__(self, world_size, discount_factor, learning_rate, epsilon):
        self.number_of_directions = 4
        self.q_table = np.zeros((world_size, world_size, self.number_of_directions))
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.goal_state = None

    def choose_action(self, current_state, action_mask):
        if np.random.rand() < self.epsilon:
            action_mask_array = np.array(action_mask)
            legal_actions = np.where(action_mask_array == 1)[0]
            return np.random.choice(legal_actions)
        else:
            actions = copy(self.q_table[current_state])
            actions[action_mask == 0] = -1
            return np.argmax(actions)

    def learn(self, current_state, action, reward, next_state):
        prediction = self.q_table[current_state, self.goal_state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, self.goal_state])
        self.q_table[current_state, self.goal_state, action] += self.learning_rate * (target - prediction)

    def set_goal(self, new_goal_state):
        self.goal_state = new_goal_state
