from environments.five_grid import FiveGrid, layouts as env_layouts
from agents.Sender import Sender
from agents.Receiver import Receiver
import numpy as np

chosen_layout = env_layouts["pong"]
env = FiveGrid(illegal_positions=chosen_layout)

# The hyperparameters
# Amount of sender agents
M_values = [1, 2, 3, 4, 5]
# Channel capacity (amount of messages each sender can send)
C_values = [3, 4, 5, 8, 9, 16, 25, 27, 32, 36, 64]
# The learning rate for RMSprop, which is an optimization algorithm used to adjust weights in the learning process.
# Listed but not used in the paper: 5e-5
eta_values = [1e-4, 5e-4, 1e-3]
# The epsilon values for the sender
# Listed but not used in the paper: 0.1, 0.15
epsilon_s_values = [0.01, 0.05]
# The epsilon values for the receiver
# Listed but not used in the paper: 0.15
epsilon_r_values = [0.01, 0.05, 0.1]
# The receiver's Q-learning discount factor
gamma_values = [0.7, 0.8, 0.9]
# The possible layouts for the environment
layouts = [env_layouts["pong"], env_layouts["four_room"], env_layouts["two_room"], env_layouts["flower"], env_layouts["empty_room"]]

# 12 million steps
learning_steps = 12000000

# Convert the state number into a one-hot vector
def state_to_one_hot(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return one_hot

def state_and_msgs_to_one_hot(state, messages, num_states, num_messages):
    one_hot = np.zeros(num_states + num_messages * len(messages))
    one_hot[state] = 1
    for i in range(len(messages)):
        one_hot[num_states + i * num_messages + messages[i]] = 1
    return one_hot


for lay_out in layouts:
    env = FiveGrid(illegal_positions=lay_out)
    for gamma in gamma_values:
        options = {"termination_probability": 1 - gamma}
        # Amount of sender agents
        for M in M_values:
            # Channel capacity (amount of messages each sender can send)
            for C in C_values:
                # The learning rate for RMSprop, which is an optimization algorithm used to adjust weights in the learning process.
                for eta in eta_values:
                    for epsilon_s in epsilon_s_values:
                        for epsilon_r in epsilon_r_values:
                            print("M: " + str(M) + ", C: " + str(C) + ", eta: " + str(eta) + ", epsilon_s: " + str(
                                epsilon_s) + ", epsilon_r: " + str(epsilon_r) + ", gamma: " + str(gamma))
                            senders = [Sender(epsilon_s, C, env.world_size, eta) for _ in range(M)]
                            receiver = Receiver(gamma, epsilon_r, env.world_size, M, C, eta)
                            action = None
                            messages = []
                            observations, infos = env.reset(options=options)
                            # Get the goal state
                            goal_state = env.returnGoal()
                            # For every sender, generate a message
                            for sender in senders:
                                context = state_to_one_hot(goal_state, env.world_size)
                                messages.append(sender.choose_action(context))
                            for step in range(learning_steps):
                                observation = observations["receiver"]
                                context = state_and_msgs_to_one_hot(observation["observation"], messages, env.world_size, C)
                                # Choose an appropriate action for the receiver
                                action = receiver.choose_action(context, observation["action_mask"])
                                # Get the observations, rewards, terminations, truncations and infos
                                next_observations, rewards, terminations, truncations, infos = env.step({"receiver": action})
                                reward = rewards["receiver"]
                                # Learn from the experience as a receiver
                                next_observation = next_observations["receiver"]
                                next_context = state_and_msgs_to_one_hot(next_observation["observation"], messages, env.world_size, C)
                                receiver.learn(context, next_context, action, reward)
                                # Check for termination or truncation
                                if terminations["receiver"] or truncations["receiver"]:
                                    # Senders all have an opportunity to learn
                                    for sender, message in zip(senders, messages):
                                        context = state_to_one_hot(goal_state, env.world_size)
                                        sender.learn(context, message, reward)
                                    # Reset the environment
                                    observations, infos = env.reset(options=options)
                                    # Get the goal state
                                    goal_state = env.returnGoal()
                                    messages = []
                                    # For every sender, generate a message
                                    for sender in senders:
                                        context = state_to_one_hot(goal_state, env.world_size)
                                        messages.append(sender.choose_action(context))
                                else:
                                    observations = next_observations





# #Generate a generic loop for the environment using an agent that does random actions
# # Path: main.py
# # Compare this snippet from agents/Receiver.py:
# def choose_action(action_mask):
#     # Choose a random action
#     action = np.random.choice(np.where(action_mask == 1)[0])
#     # Return the action
#     return action
#
# # Termination probability of the environment
# options = {"termination_probability": 0.1}
#
# observations, infos = env.reset(options=options)
#
# env.render()
#
# while env.agents:
#     # From observations, get the action mask
#     action_mask = observations["receiver"]["action_mask"]
#     choice = choose_action(action_mask)
#     # Get the observations, rewards, terminations, truncations and infos
#     observations, rewards, terminations, truncations, infos = env.step({"receiver": choice})
#     # Print receiver reward
#     print("Receiver reward: " + str(rewards["receiver"]))
#     # Render the environment
#     env.render()

