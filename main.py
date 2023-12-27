from environments.five_grid import FiveGrid, layouts as env_layouts
from agents.Sender import Sender
from agents.Receiver import Receiver
from agents.QLearningAgent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from plotting import plot_reward_curves

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
learning_steps = 4000

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

def run_q_agent(gamma, epsilon, learning_rate, env, learning_steps):
    options = {"termination_probability": 1 - gamma}
    receiver = QLearningAgent(env.world_size, gamma, learning_rate, epsilon)
    action = None
    observations, infos = env.reset(options=options)
    goal_state = env.returnGoal()
    receiver.set_goal(goal_state)

    episodes_rewards = []
    episode_steps = []
    episode_total_steps = []

    # Create a progress bar for the learning steps
    progress_bar = tqdm(total=learning_steps, position=0, leave=True)

    for step in range(learning_steps):
        current_observation_and_mask = observations["receiver"]
        #print("current_observation_and_mask: " + str(current_observation_and_mask))
        current_observation = current_observation_and_mask["observation"]
        #print("current_observation: " + str(current_observation))
        current_mask = current_observation_and_mask["action_mask"]
        #print("current_mask: " + str(current_mask))
        action = receiver.choose_action(current_observation, current_mask)
        #print("action: " + str(action))
        next_observations, rewards, terminations, truncations, infos = env.step({"receiver": action})
        reward = rewards["receiver"]
        #print("reward: " + str(reward))
        next_observation_and_mask = next_observations["receiver"]
        #print("next_observation_and_mask: " + str(next_observation_and_mask))
        next_observation = next_observation_and_mask["observation"]
        #print("next_observation: " + str(next_observation))
        receiver.learn(current_observation, action, reward, next_observation)

        if terminations["receiver"] or truncations["receiver"]:
            episode_steps.append(env.timestep)
            episodes_rewards.append(reward)
            episode_total_steps.append(step + 1)
            observations, infos = env.reset(options=options)
            goal_state = env.returnGoal()
            receiver.set_goal(goal_state)
            #print("end of episode")
        else:
            observations = next_observations
        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_description(f"Step {step}/{learning_steps}")

    # Close the progress bar
    progress_bar.close()

    return episodes_rewards, episode_steps, episode_total_steps

    # Any additional logic or cleanup

def find_sender_message_combinations(C):
    """
    Find all combinations of senders and messages where the number of channels (C) is equal to N^M.
    """
    combinations = []
    # We check up to C because we can have a maximum of C senders each sending 1 message.
    for N in range(1, C + 1):
        # M = C^(1/N) -> Check if M is an integer
        M = C ** (1 / N)
        if M.is_integer():
            combinations.append((N, int(M)))
    return combinations


def run_experiment(M, num_messages, eta, epsilon_s, epsilon_r, gamma, env, learning_steps):
    # C = num_messages ** M
    print(f"M: {M}, Number Of Possible Messages: {num_messages}, eta: {eta}, epsilon_s: {epsilon_s}, epsilon_r: {epsilon_r}, gamma: {gamma}")
    options = {"termination_probability": 1 - gamma}
    senders = [Sender(epsilon_s, num_messages, env.world_size, eta) for _ in range(M)]
    receiver = Receiver(gamma, epsilon_r, env.world_size, M, num_messages, eta)
    action = None
    messages = []
    observations, infos = env.reset(options=options)
    goal_state = env.returnGoal()

    for sender in senders:
        context = state_to_one_hot(goal_state, env.world_size)
        messages.append(sender.choose_action(context))

    episodes_rewards = []
    episode_steps = []
    episode_total_steps = []

    # Create a progress bar for the learning steps
    progress_bar = tqdm(total=learning_steps, position=0, leave=True)

    for step in range(learning_steps):
        observation = observations["receiver"]
        context = state_and_msgs_to_one_hot(observation["observation"], messages, env.world_size, num_messages)
        action = receiver.choose_action(context, observation["action_mask"])
        next_observations, rewards, terminations, truncations, infos = env.step({"receiver": action})
        reward = rewards["receiver"]
        next_observation = next_observations["receiver"]
        next_context = state_and_msgs_to_one_hot(next_observation["observation"], messages, env.world_size, num_messages)
        receiver.add_example(context, next_context, action, reward)
        if step % 10 == 0:
            receiver.learn()

        if terminations["receiver"] or truncations["receiver"]:
            for sender, message in zip(senders, messages):
                context = state_to_one_hot(goal_state, env.world_size)
                sender.learn(context, message, reward)
            episode_steps.append(env.timestep)
            episodes_rewards.append(reward)
            episode_total_steps.append(step + 1)
            observations, infos = env.reset(options=options)
            goal_state = env.returnGoal()
            messages = []
            for sender in senders:
                context = state_to_one_hot(goal_state, env.world_size)
                messages.append(sender.choose_action(context))
        else:
            observations = next_observations

        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_description(f"Step {step}/{learning_steps}")

    # Close the progress bar
    progress_bar.close()

    return episodes_rewards, episode_steps, episode_total_steps

    # Any additional logic or cleanup

# Generate a plot for the rewards and steps

# Plot the results
env = FiveGrid(illegal_positions=chosen_layout)
gamma = 0.7
#rewards, steps, total_steps = run_experiment(1, 4, 0.001, 0.01, 0.01, gamma, env, learning_steps)
#rewards, steps, total_steps = run_q_agent(gamma, 0.01, 0.9, env, learning_steps)
# Generate a plot for the rewards and steps

# for lay_out in layouts:
#     env = FiveGrid(illegal_positions=lay_out)
#     for gamma in gamma_values:
#         options = {"termination_probability": 1 - gamma}
#         # Amount of sender agents
#         for M in M_values:
#             # Channel capacity (amount of messages each sender can send)
#             for C in C_values:
#                 # The learning rate for RMSprop, which is an optimization algorithm used to adjust weights in the learning process.
#                 for eta in eta_values:
#                     for epsilon_s in epsilon_s_values:
#                         for epsilon_r in epsilon_r_values:
#                             run_experiment(M, C, eta, epsilon_s, epsilon_r, gamma, env, learning_steps)





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

