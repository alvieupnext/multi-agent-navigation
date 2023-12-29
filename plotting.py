import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_reward_curves(pdDataframe):
    """
    For each of the environments, plot the reward curves for the different settings.
    """
    environments = ["pong", "four_room", "two_room", "flower", "empty_room"]
    for env in environments:
        plot_reward_curve(pdDataframe[pdDataframe["Env"] == env], env)


gamma = 0.8


def plot_reward_curve(pdDataframe, environment):
    """
    Function that plot the reward curves for the different types of settings for a given environment.
    The dataframe has the episodes as rows, and the rewards, steps, total_steps, channel capacity, setting, and the environment as columns.
    The episodes will be a positive integer. The rewards are either 0 or 1. The steps are a positive integer. The total_steps are
    a positive integer. The channel capacity is a positive integer. The setting is a string. The environment is a string.
    The different settings are the following: 5S-1R, 4S-1R, 3S-1R, 2S-1R, 1S-1R, Random, Q-learning, Max.
    The different environments are the following: Pong, Four-room, Two-room, Flower, Empty-room.
    """
    settings = ["5S-1R", "4S-1R", "3S-1R", "2S-1R", "1S-1R", "Random", "Q-learning", "Max"]
    # Create a figure
    fig, ax = plt.subplots()

    # For each of the settings, plot the reward curve. Rewards are plotted as a sliding window average over 1000 steps
    for setting in settings:
        data = pdDataframe[pdDataframe["Type"] == setting]
        # Group the data by the total steps
        # 10 -> [[], []]
        rewards = data["Reward"] * gamma ** data["Steps"]
        total_steps = data["TotalSteps"]
        # Plot the rewards
        ax.plot(total_steps, rewards.rolling(50000).mean(), label=setting)

    # Set the title
    ax.set_title(f"Reward curves for {environment}")
    # Set the x-axis label
    ax.set_xlabel("Steps")
    # Set the y-axis label
    ax.set_ylabel("Reward")
    # Set the legend
    ax.legend()
    # Show the plot
    plt.show()


# # Load tabular_results_pong_0.0001_0.05_0.05_0.8_12000000.csv from the results folder as a pandas dataframe
# df = pd.read_csv("results/tabular_results_pong_0.0001_0.05_0.05_0.8_12000000.csv")
# # Plot the reward curves
# plot_reward_curve(df, "pong")


#
# Generate mock data for testing the function
#

# # Generate mock data for testing the function
# np.random.seed(0)  # For reproducibility
#
# # Define parameters for mock data
# num_episodes = 10_000  # Number of episodes
# environments = ["pong", "four_room", "two_room", "flower", "empty_room"]
# settings = ['5S-1R', '4S-1R', '3S-1R', '2S-1R', '1S-1R', 'Random', 'Q-learning', 'Max']
#
# # Create an empty DataFrame
# df = pd.DataFrame()
#
# # Populate the DataFrame
# for env in environments:
#     for setting in settings:
#         # Generate a linearly increasing reward with some noise
#         reward = np.linspace(0, 1, num_episodes) + np.random.normal(0, 0.1, num_episodes)
#         reward = np.clip(reward, 0, 1)  # Ensure rewards are between 0 and 1
#
#         # Generate other columns
#         episodes = np.arange(0, num_episodes)
#         steps = np.random.randint(1, 100, num_episodes)
#         total_steps = np.cumsum(steps)
#         channel_capacity = np.random.randint(1, 10, num_episodes)
#
#         # Create a temp DataFrame and append to the main DataFrame
#         temp_df = pd.DataFrame({
#             'Episodes': episodes,
#             'Reward': reward,
#             'Steps': steps,
#             'TotalSteps': total_steps,
#             'C': channel_capacity,
#             'Type': setting,
#             'Env': env
#         })
#         df = pd.concat([df, temp_df])
#
# # Reset index
# df = df.reset_index(drop=True)

# # Plot the reward curves
# plot_reward_curves(df)

def visualize_belief(belief_table):
    indices = np.argmax(belief_table, axis=1)
    indices = indices.reshape((5, 5))
    # display the value in each cell
    for i in range(5):
        for j in range(5):
            plt.text(j, i, indices[i, j], ha='center', va='center', color='black')
    plt.imshow(indices)
    plt.colorbar()
    plt.show()
