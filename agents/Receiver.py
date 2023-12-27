import tensorflow as tf
import numpy as np


class Receiver:

    def __init__(self, discount_factor, epsilon, world_size, num_senders, channel_capacity, eta):
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.observation_size = world_size+num_senders*channel_capacity
        self.number_of_directions = 4
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=eta)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.number_of_directions, activation='softmax', input_shape=(self.observation_size,))
        ])
        self.model.compile(optimizer=optimizer, loss='mse')
        # Buffer that will contain the examples seen after each step of an episode. At the end of the episode, the
        # examples are taken from the buffer and used to update the model in batch.
        self.buffer = []

    def choose_action(self, observation, action_mask):

        # If a random number is smaller than epsilon => take a random direction (exploration)
        if np.random.rand() < self.epsilon:
            action_mask_array = np.array(action_mask)
            legal_actions = np.where(action_mask_array == 1)[0]
            return np.random.choice(legal_actions)
        # Otherwise, follow the model to predict the next move of the agent (exploitation)
        else:
            prediction = self.model.predict(np.reshape(observation, (1, self.observation_size)),verbose=0)
            prediction = np.reshape(prediction, (self.number_of_directions,))
            # Set the probability of the unavailable actions to -inf
            prediction[action_mask == 0] = 0
            return np.argmax(prediction)

    def learn(self):
        """
        Updates the model with the examples in the buffer
        """
        # Get the observations and outputs from the buffer
        observations, outputs = zip(*self.buffer)
        # Concatenate the observations and outputs
        observations = np.concatenate(observations)
        outputs = np.concatenate(outputs)
        # Train the model on the batch
        self.model.fit(observations, outputs, verbose=0)
        # Empty the buffer
        self.buffer = []

    def add_example(self, observation, next_observation, action, reward):
        """
        Adds an example to the buffer
        """
        output = np.zeros(self.number_of_directions)
        max_q = np.argmax(self.model.predict(np.reshape(next_observation, (1, self.observation_size)), verbose=0))
        output[action] = reward + (self.discount_factor * max_q)
        # Reshape the context vector
        observation = np.reshape(observation, (1, self.observation_size))
        # Reshape the output vector
        output = np.reshape(output, (1, self.number_of_directions))
        self.buffer.append((observation, output))

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
