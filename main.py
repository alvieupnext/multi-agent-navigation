from environments.five_grid import FiveGrid, layouts as env_layouts
from agents.Sender import Sender
from agents.Receiver import Receiver
import numpy as np
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


def run_experiment(M, C, eta, epsilon_s, epsilon_r, gamma, env, learning_steps):
    print(f"M: {M}, C: {C}, eta: {eta}, epsilon_s: {epsilon_s}, epsilon_r: {epsilon_r}, gamma: {gamma}")
    options = {"termination_probability": 1 - gamma}
    senders = [Sender(epsilon_s, C, env.world_size, eta) for _ in range(M)]
    receiver = Receiver(gamma, epsilon_r, env.world_size, M, C, eta)
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
        context = state_and_msgs_to_one_hot(observation["observation"], messages, env.world_size, C)
        action = receiver.choose_action(context, observation["action_mask"])
        next_observations, rewards, terminations, truncations, infos = env.step({"receiver": action})
        reward = rewards["receiver"]
        next_observation = next_observations["receiver"]
        next_context = state_and_msgs_to_one_hot(next_observation["observation"], messages, env.world_size, C)
        receiver.add_example(context, next_context, action, reward)
        if step % 10 == 0:
            receiver.learn()

        if terminations["receiver"] or truncations["receiver"]:
            for sender, message in zip(senders, messages):
                context = state_to_one_hot(goal_state, env.world_size)
                sender.learn(context, message, reward)
            episode_steps.append(env.timestep)
            episodes_rewards.append(reward)
            episode_total_steps.append(step)
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

env = FiveGrid(illegal_positions=chosen_layout)
gamma = 0.7
rewards, steps, total_steps = run_experiment(1, 4, 0.001, 0.01, 0.01, gamma, env, learning_steps)

# Generate a plot for the rewards and steps
plt.plot(rewards)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.show()

plt.plot(steps)
plt.ylabel('Steps')
plt.xlabel('Episode')
plt.show()

plt.plot(total_steps)
plt.ylabel('Total steps')
plt.xlabel('Episode')
plt.show()

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

