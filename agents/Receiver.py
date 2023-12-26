import tensorflow as tf
import numpy as np


class Receiver:

    def __init__(self, discount_factor, epsilon, world_size, num_senders, channel_capacity):
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.observation_size = world_size+num_senders*channel_capacity
        self.number_of_directions = 4
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.number_of_directions, activation='softmax', input_shape=(self.observation_size,))
        ])
        self.model.compile(optimizer='rmsprop', loss='mse')

    def choose_action(self, observation, action_mask):

        # If a random number is smaller than epsilon => take a random direction (exploration)
        if np.random.rand() < self.epsilon:
            print("random direction")
            return np.random.randint(self.number_of_directions)
        # Otherwise, follow the model to predict the next move of the agent (exploitation)
        else:
            print("following the model's direction")
            prediction = self.model.predict(np.reshape(observation, (1, self.observation_size)))
            prediction[action_mask == 0] = -np.inf
            return np.argmax(prediction)

    def learn(self, observation, next_observation, action, reward):
        output = np.zeros(self.number_of_directions)
        max_q = np.argmax(self.model.predict(np.reshape(next_observation, (1, self.observation_size))))
        output[action] = reward + (self.discount_factor * max_q)
        # Reshape the context vector
        observation = np.reshape(observation, (1, self.observation_size))
        # Reshape the output vector
        output = np.reshape(output, (1, self.number_of_directions))
        # Update the model
        self.model.fit(observation, output, verbose=0)

    def update_epsilon(self):
        pass


if __name__ == "__main__":
    receiver = Receiver(0.1, 0.9, 25, 1, 3)
    # in a for loop, generate a random context vector, choose an action, learn. Print the weights of the model
    for i in range(100):
        observation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        action_mask = [0, 1, 1, 0]
        action = receiver.choose_action(observation, action_mask)
        new_observation = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        reward = 0
        receiver.learn(observation, new_observation, action, reward)
        print(receiver.model.get_weights())
