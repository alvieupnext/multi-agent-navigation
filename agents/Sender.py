import tensorflow as tf
import numpy as np


class Sender:
    """
    A sender agent is modelled as a contextual N-armed bandit that selects one message action m out of N possible messages.
    This is based on the static context vector c, The context is a one-hot vector encoding the goal location in the flattened
    array representing the gridworld. The length of the vector is equal to the size of the gridworld.

    The i-th sender’s action-value estimation function Q(·) is implemented as a single layer feed-forward neural network
    parametrized by theta_(si). The number of neurons in the output layer is equal to the number of possible messages.
    The number of neurons in the input layer is equal to the size of the gridworld. The input layer is fed with
    the context vector c.

    The loss for a single sender is L_(si) = (R_t - Q(c, m_i; theta_(si)))^2, where R_t is the reward received
    at the end of an episode of length T with the value of 1 or 0 depending on whether the receiver reached
    the goal state, c is the context and m_i is a message action.

    Message actions are selected using an ε-greedy policy.
    """

    def __init__(self, epsilon, num_possible_messages, world_size, eta):
        self.epsilon = epsilon
        self.num_possible_messages = num_possible_messages
        self.world_size = world_size
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=eta)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_possible_messages, activation='softmax', input_shape=(world_size,))
        ])
        self.model.compile(optimizer=optimizer, loss='mse')

    def choose_action(self, context):
        """
        Choose an action based on the context and the epsilon greedy policy
        :param context: the context vector
        :return: the chosen action
        """
        # Generate a random number and check if it is less than epsilon, if this is the case, choose a random message action, use numpy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_possible_messages)
        # Otherwise, use the model to predict the message action
        else:
            return np.argmax(self.model.predict(np.reshape(context, (1, self.world_size))))

    def learn(self, context, message_action, reward):
        """
        Update the model based on the reward
        """
        # Create the output vector where all values are 0 except for the chosen message action that has the value of the reward
        output = np.zeros(self.num_possible_messages)
        output[message_action] = reward
        # Reshape the context vector
        context = np.reshape(context, (1, self.world_size))
        # Reshape the output vector
        output = np.reshape(output, (1, self.num_possible_messages))
        # Update the model
        self.model.fit(context, output, verbose=0)

# test
if __name__ == "__main__":
    sender = Sender(0.9, 4, 25)
    # in a for loop, generate a random context vector, choose an action, learn. Print the weights of the model
    for i in range(1000):
        context = np.random.randint(2, size=25)
        action = sender.choose_action(context)
        reward = np.random.randint(2)
        sender.learn(context, action, reward)
        print(sender.model.get_weights())
