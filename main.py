from environments.five_grid import FiveGrid, layouts as env_layouts
from agents.Sender import Sender
from agents.Receiver import Receiver
from agents.QLearningAgent import QLearningAgent
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Flatten an array of messages into a single message (one hot encoding)
def flatten_messages(messages, num_messages):
    return sum([message * num_messages ** i for i, message in enumerate(messages)])

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
    # progress_bar = tqdm(total=learning_steps, position=0, leave=True)

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
        # progress_bar.update(1)
        # progress_bar.set_description(f"Q-Learning Step {step}/{learning_steps}")

    # Close the progress bar
    # progress_bar.close()

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


epsilon_max = 1
epsilon_min = 0.01
epsilon_decay = 0.99995
def run_experiment(M, num_messages, alpha, epsilon_s, epsilon_r, gamma, env, learning_steps):
    # C = num_messages ** M
    print(f"M: {M}, Number Of Possible Messages: {num_messages}, alpha: {alpha}, epsilon_s: {epsilon_s}, epsilon_r: {epsilon_r}, gamma: {gamma}")
    options = {"termination_probability": 1 - gamma}
    senders = [Sender(epsilon_max, epsilon_min, epsilon_decay, num_messages, env.world_size, alpha) for _ in range(M)]
    receiver = Receiver(env.world_size, num_messages ** M, gamma, alpha, epsilon_max, epsilon_min, epsilon_decay)
    action = None
    messages = []
    observations, infos = env.reset(options=options)
    goal_state = env.returnGoal()

    for sender in senders:
        messages.append(sender.choose_action(goal_state))

    # Flatten the messages into a single message
    message = flatten_messages(messages, num_messages)

    episodes_rewards = []
    episode_steps = []
    episode_total_steps = []

    # Create a progress bar for the learning steps
    # progress_bar = tqdm(total=learning_steps, position=0, leave=True)

    for step in range(learning_steps):
        observation = observations["receiver"]
        action = receiver.choose_action(observation["observation"], message, observation["action_mask"])
        next_observations, rewards, terminations, truncations, infos = env.step({"receiver": action})
        reward = rewards["receiver"]
        next_observation = next_observations["receiver"]
        receiver.learn(observation["observation"], message,action, reward, next_observation["observation"])

        if terminations["receiver"] or truncations["receiver"]:
            for sender, message in zip(senders, messages):
                sender.learn(goal_state, message, reward)
                sender.update_epsilon()
            receiver.update_epsilon()
            episode_steps.append(env.timestep)
            episodes_rewards.append(reward)
            episode_total_steps.append(step + 1)
            observations, infos = env.reset(options=options)
            goal_state = env.returnGoal()
            messages = []
            for sender in senders:
                messages.append(sender.choose_action(goal_state))
        else:
            observations = next_observations

    # Check if the sender and receiver have learned the correct policy
    for sender in senders:
        print(f"Sender Q-Table: {sender.belief_table}")
        print(f"Receiver Q-Table: {receiver.q_table}")

        # Update the progress bar
        # progress_bar.update(1)
        # progress_bar.set_description(f"Step {step}/{learning_steps}, M: {M}, Number Of Possible Messages: {num_messages}")

    # Close the progress bar
    # progress_bar.close()

    return episodes_rewards, episode_steps, episode_total_steps

    # Any additional logic or cleanup

# Generate a plot for the rewards and steps

# Plot the results
env = FiveGrid(illegal_positions=chosen_layout)
gamma = 0.8
episodes_rewards, episode_steps, episode_total_steps = run_experiment(1, 4, 0.001, 0.05, 0.05, gamma, env, learning_steps)
import numpy as np
#Calculate rolling average with a window of 10000
rolling_average = np.convolve(episodes_rewards, np.ones((50000,))/50000, mode='valid')
plt.plot(rolling_average)

plt.xlabel("Learning Steps")
plt.ylabel("Reward")
plt.title(f"Reward vs Learning Steps (Gamma: {gamma})")
plt.show()


# rewards, steps, total_steps = run_experiment(1, 4, 0.001, 0.01, 0.01, gamma, env, learning_steps)
# #rewards, steps, total_steps = run_q_agent(gamma, 0.01, 0.9, env, learning_steps)
# # Generate a plot for the rewards and steps
# plt.plot(total_steps, rewards)
# plt.xlabel("Learning Steps")
# plt.ylabel("Reward")
# plt.title(f"Reward vs Learning Steps (Gamma: {gamma})")
# plt.show()

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

