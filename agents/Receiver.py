import numpy as np


class Receiver:

    def __init__(self, world_size, channel_capacity, discount_factor, learning_rate, epsilon):
        self.number_of_directions = 4
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.channel_capacity = channel_capacity
        self.q_table = np.zeros((world_size, self.channel_capacity, self.number_of_directions))


    def choose_action(self, current_state, message, action_mask):
        if np.random.rand() < self.epsilon:
            action_mask_array = np.array(action_mask)
            legal_actions = np.where(action_mask_array == 1)[0]
            return np.random.choice(legal_actions)
        else:
            actions = self.q_table[current_state, message]
            actions[action_mask == 0] = -1
            return np.argmax(actions)

    def learn(self, current_state, message, action, reward, next_state):
        prediction = self.q_table[current_state, message, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, message])
        self.q_table[current_state, message, action] += self.learning_rate * (target - prediction)
