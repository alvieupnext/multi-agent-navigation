import numpy as np


class Sender:
    """
    A sender agent is modelled as a contextual N-armed bandit that selects one message action m out of N possible messages.
    This is based on the static context vector c, The context is a one-hot vector encoding the goal location in the flattened
    array representing the gridworld. The length of the vector is equal to the size of the gridworld.

    Learning the optimal message actions is done using a Q-learning algorithm. The Q-values are stored in a Q-table.

    Message actions are selected using an ε-greedy policy.
    """

    def __init__(self, epsilon, num_possible_messages, world_size, alpha):
        self.epsilon = epsilon
        self.num_possible_messages = num_possible_messages
        self.world_size = world_size
        self.alpha = alpha
        self.q_table = np.zeros((world_size, num_possible_messages))

    def choose_action(self, context):
        """
        Choose an action based on the context and the epsilon greedy policy
        :param context: the context vector
        :return: the chosen action
        """
        # Generate a random number and check if it is less than epsilon, if this is the case, choose a random message action, use numpy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_possible_messages)
        # Otherwise, use the Q-table to choose the message action with the highest Q-value
        else:
            return np.argmax(self.q_table[context])

    def learn(self, context, message_action, reward):
        """
        Update the Q-table based on the reward
        """
        # Get the Q-value for the current context and message action
        prediction = self.q_table[context, message_action]
        # Get the Q-value for the next context and the best message action
        target = reward + np.max(self.q_table[context])
        # Update the Q-table
        self.q_table[context, message_action] += self.alpha * (target - prediction)
