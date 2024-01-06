import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from scipy.stats import beta
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# def plot_reward_curves(pdDataframe):
#     """
#     For each of the environments, plot the reward curves for the different settings.
#     """
#     # environments = ["pong", "four_room", "two_room", "flower", "empty_room"]
#     for env in environments:
#         plot_reward_curve(pdDataframe[pdDataframe["Env"] == env], env)
#
#
# environments = ["pong", "four_room", "two_room", "flower", "empty_room"]
#
# gamma = 0.8
#
#
# def plot_reward_curve(pdDataframe, environment):
#     settings = ["5S-1R", "4S-1R", "3S-1R", "2S-1R", "1S-1R", "Random", "Q-learning", "Max"]
#     fig, ax = plt.subplots()
#
#     for setting in settings:
#         data = pdDataframe[pdDataframe["Type"] == setting]
#
#         # Group by TotalSteps and get for each group the mean and standard deviation of the DiscountedReward
#         grouped_data = data.groupby("TotalSteps")["DiscountedReward"].agg(["mean"]).reset_index()
#
#         # print(grouped_data["mean"])
#         # print(grouped_data["std"])
#
#         # Calculate the rolling mean and standard deviation
#         rolling_mean = grouped_data["mean"].rolling(window=50000).mean()
#         # rolling_std = grouped_data["std"].rolling(window=500).mean()
#         # Plot mean and error zone
#         ax.plot(grouped_data['TotalSteps'], rolling_mean, label=setting)
#         # ax.fill_between(grouped_data['TotalSteps'], rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=1)
#
#     ax.set_title(f"Reward curves with error zones for {environment}")
#     ax.set_xlabel("Steps")
#     ax.set_ylabel("Reward")
#     ax.legend()
#     plt.show()

# Theoretical Max Reward (calculated by the average max discounted reward that can be reached by a perfect agent
# in the environment)
# max_rewards = {"pong": 0.56045714285, "four_room": 0.54968888888,
#                "two_room": 0.55872, "flower": 0.57472, "empty_room": 0.5856}

# max_rewards = {"pong": 0.56045714285, "four_room": 0.54968888888,
#                "two_room": 0.55872, "flower": 0.57472, "empty_room": 0.5856}

max_rewards_8 = {"pong": 0.56045714285, "four_room": 0.54968888888,
               "two_room": 0.55872, "flower": 0.57472, "empty_room": 0.5856}

max_rewards_9 = {"pong": 0.75574285714, "four_room": 0.7498, "two_room": 0.75582, "flower": 0.76482, "empty_room": 0.77235}


def plot_reward_curves(pdDataframe, environments=None):
    """
    For each of the environments, plot the reward curves for the different settings in subplots.
    """
    if environments is None:
        environments = ["pong", "four_room", "two_room", "flower", "empty_room"]
    settings = ["5S-1R", "4S-1R", "3S-1R", "2S-1R", "1S-1R", "Random", "Q-learning"]
    fig, axes = plt.subplots(1, len(environments), figsize=(20, 4))  # Adjust the figsize as needed
    lines = []  # To keep track of line objects for the legend

    # Plot each environment in a subplot
    for idx, env in enumerate(environments):
        ax = axes[idx]
        data = pdDataframe[pdDataframe["Env"] == env]

        for setting in settings:
            subset = data[data["Type"] == setting]
            grouped_data = subset.groupby("TotalSteps")["DiscountedReward"].agg(["mean"]).reset_index()
            rolling_mean = grouped_data["mean"].rolling(window=50000).mean()
            line, = ax.plot(grouped_data['TotalSteps'], rolling_mean, label=setting)
            if env == environments[0]:  # Only add lines once for the legend
                lines.append(line)

        # Plot the theoretical max reward
        line = ax.axhline(y=max_rewards_9[env], color='r', linestyle='--', label="Max")
        if env == environments[0]:  # Only add lines once for the legend
            lines.append(line)


        ax.set_title(f"{env.capitalize()}")
        ax.set_xlabel("Steps")
        if idx == 0:  # Only the first subplot gets the ylabel to avoid clutter
            ax.set_ylabel("Reward")
    # Create a single legend for all subplots at the top
    fig.legend(lines, settings + ["Max"], loc='upper center', ncol=len(settings)+1)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig("Fig3A.pgf")

    plt.show()

def plot_average_return_per_channel_capacity_excluding(pdDataframe, environments, exclude_settings):
    fig, axs = plt.subplots(1, len(environments), figsize=(20, 5), sharey=True)

    # Filter out the settings we want to exclude
    filtered_data = pdDataframe[~pdDataframe['Type'].isin(exclude_settings)]

    for ax, environment in zip(axs, environments):
        # Filter data for the current environment
        env_data = filtered_data[filtered_data["Env"] == environment]

        # Group by ChannelCapacity and calculate average return
        grouped_data = env_data.groupby('C')['DiscountedReward'].mean().reset_index()

        # Plotting the average return for the current environment
        ax.plot(grouped_data.index, grouped_data['DiscountedReward'], marker='o', linestyle='-',
                label="Average excluding specified settings")  # Using index for equal spacing

        ax.set_title(environment.capitalize())
        ax.set_xlabel("Channel Capacity")

        # Set the x ticks to be equally spaced and label them with the channel capacities
        ax.set_xticks(grouped_data.index)
        ax.set_xticklabels(grouped_data['C'])

        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Reward*")

    plt.tight_layout()
    plt.savefig("Fig3B.pgf")
    plt.show()

def plot_sorted_drop_with_confidence(mean_drops, confidence_intervals):
    """
    Plots a bar graph with the mean drops (sorted in descending order) and their confidence intervals.
    Adjusted to address deprecation warnings and the ValueError for colorbar.

    Parameters:
    mean_drops (np.array): An array containing the mean drop in performance for each sender.
    confidence_intervals (np.array): An array of tuples containing the lower and upper bounds of the confidence intervals.
    """
    # Ensure that the arrays are numpy arrays
    mean_drops = np.array(mean_drops)
    confidence_intervals = np.array(confidence_intervals)

    # Sort the drops and confidence intervals together in descending order
    sorted_indices = np.argsort(mean_drops)[::-1]
    sorted_mean_drops = mean_drops[sorted_indices]
    sorted_confidence_intervals = confidence_intervals[sorted_indices]

    # Calculate the errors from the confidence intervals
    errors = np.abs(np.column_stack((sorted_mean_drops - sorted_confidence_intervals[:, 0],
                                     sorted_confidence_intervals[:, 1] - sorted_mean_drops))).T

    # Normalize the mean drops to get a range for color mapping
    normalized_drops = (sorted_mean_drops - np.min(sorted_mean_drops)) / np.ptp(sorted_mean_drops)

    # Create the colormap using the updated syntax
    cmap = plt.cm.get_cmap('Blues')

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(np.arange(len(sorted_mean_drops)), sorted_mean_drops, color=cmap(normalized_drops + 0.3), yerr=errors,
                  capsize=5)

    # Create a ScalarMappable and initialize a data structure
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=np.min(sorted_mean_drops), vmax=np.max(sorted_mean_drops)))
    sm.set_array([])  # You need to set_array for the ScalarMappable.

    # Add the color bar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Drop in performance (%)', rotation=270, labelpad=15)

    # Add labels and title
    ax.set_xlabel('Sorted sender ID')
    ax.set_ylabel('Drop in performance (%)')
    ax.set_xticks(np.arange(len(sorted_mean_drops)))
    ax.set_xticklabels(sorted_indices)  # Set x-ticks to be the sorted sender IDs
    plt.savefig("Fig6.pgf")

    # Show the plot
    plt.show()


# print("before reading")
# Open tabular_results_pong_0.0001_0.05_0.05_0.8_12000000 from results folder
# df = pd.read_csv("results/tabular_results_pong_0.0001_0.05_0.05_0.8_12000000.csv")
# print("after reading")

# Plot the reward curves
# plot_reward_curve(df, "pong")

gamma = 0.9
# Load the data from the results folder
pong = pd.read_csv("results/tabular_results_pong_0.0001_1_0.001_0.9999951365_0.9_4000000.csv", index_col=0)
four_room = pd.read_csv("results/tabular_results_four_room_0.0001_1_0.001_0.9999951365_0.9_4000000.csv", index_col=0)
two_room = pd.read_csv("results/tabular_results_two_room_0.0001_1_0.001_0.9999951365_0.9_4000000.csv", index_col=0)
flower = pd.read_csv("results/tabular_results_flower_0.0001_1_0.001_0.9999951365_0.9_4000000.csv", index_col=0)
empty_room = pd.read_csv("results/tabular_results_empty_room_0.0001_1_0.001_0.9999951365_0.9_4000000.csv", index_col=0)
# These dataframes have the same columns, so we can concatenate them
df = pd.concat([pong,
                four_room,
                two_room,
                flower, empty_room
                ]
               )
print(df.head())
df['DiscountedReward'] = df['Reward'] * gamma ** df['Steps']
environments = ["pong", "four_room", "two_room", "flower", "empty_room"]
# # Plot the reward curves
plot_reward_curves(df)
# Per environment, assign the maximum possible reward based on the environment
df['MaxReward'] = df['Env'].map(max_rewards_9)
# Divide the discounted reward by the maximum possible reward to get the percentage of the maximum reward
df['DiscountedReward'] = df['DiscountedReward'] / df['MaxReward']
plot_average_return_per_channel_capacity_excluding(df, environments, ["Random", "Q-learning", "Max"])



#
# Generate mock data for testing the function
#

# Generate mock data for testing the function
# np.random.seed(0)  # For reproducibility

# Define parameters for mock data
num_episodes = 10_000  # Number of episodes
# settings = ['5S-1R']
channel_capacity = [3, 4, 5, 8, 9, 16, 25, 27, 32, 64]

# # Create an empty DataFrame
# df = pd.DataFrame()
#
# # Populate the DataFrame
# for env in environments:
#     for setting in settings:
#         for c in channel_capacity:
#             # Generate a linearly increasing reward with some noise
#             reward = np.linspace(0, 1, num_episodes) + np.random.normal(0, 0.1, num_episodes)
#             reward = np.clip(reward, 0, 1)  # Ensure rewards are between 0 and 1
#
#             # Generate other columns
#             episodes = np.arange(0, num_episodes)
#             steps = np.random.randint(1, 100, num_episodes)
#             total_steps = np.cumsum(steps)
#
#             # Create a temp DataFrame and append to the main DataFrame
#             temp_df = pd.DataFrame({
#                 'Episodes': episodes,
#                 'Reward': reward,
#                 'Steps': steps,
#                 'TotalSteps': total_steps,
#                 'C': c,
#                 'Type': setting,
#                 'Env': env
#             })
#             df = pd.concat([df, temp_df])
#
# # Reset index
# df = df.reset_index(drop=True)

# Load tabular_results_pong_0.0001_1_0.001_0.9999951365_0.8_4000000
# df = pd.read_csv("results/tabular_results_pong_0.0001_1_0.001_0.9999951365_0.8_4000000.csv", index_col=0)
#
# print(df.head())
#
# # Calculate the discounted rewards and add them to the DataFrame

#
#
# # Plot the reward curves
# plot_reward_curve(df, "pong")
#
# df['DiscountedReward'] = df['DiscountedReward']/0.71
#

def visualize_belief(belief_table, layout):
    print(belief_table)
    indices = np.argmax(belief_table, axis=1)
    # get max index
    wall = np.max(indices) + 1
    indices = indices.reshape((5, 5))
    # display the value in each cell
    for i in range(5):
        for j in range(5):
            if (j, i) in layout:
                indices[i, j] = wall
            else:
                plt.text(j, i, indices[i, j], ha='center', va='center', color='black')
    plt.imshow(indices)
    plt.colorbar()
    plt.show()


def visualize_thompson(alphas, betas, num_messages, layout):
    print(f"Alphas: {alphas}")
    print(f"Betas: {betas}")
    messages = np.zeros(25)
    for i in range(25):
        samples = [beta.rvs(alphas[i][a], betas[i][a]) for a in range(num_messages)]
        messages[i] = np.argmax(samples)
    messages = messages.reshape((5, 5))
    wall = np.max(messages) + 1
    # display the value in each cell
    for i in range(5):
        for j in range(5):
            if (j, i) in layout:
                messages[i, j] = wall
            else:
                plt.text(j, i, messages[i, j], ha='center', va='center', color='black')
    plt.imshow(messages)
    plt.colorbar()
    plt.show()


def visualize_receiver_policy(q_table, world_size, channel_capacity, message, layout):
    dimensions = int(world_size ** (1 / 2))
    policy_table = np.argmax(q_table[:, message, :], axis=1).reshape((dimensions, dimensions))
    directions = {
        0: (0, -0.3),  # Up
        1: (0, 0.3),  # Down
        2: (0.3, 0),  # Right
        3: (-0.3, 0)  # Left
    }
    wall = 4
    print(q_table)
    for i in range(dimensions):
        for j in range(dimensions):
            if (j, i) in layout:
                policy_table[i, j] = wall
            else:
                dx, dy = directions[policy_table[i, j]]
                plt.arrow(j - dx, i - dy, dx * 2, dy * 2, head_width=0.1, head_length=0.1, fc='k', ec='k')
    plt.imshow(policy_table)
    plt.title("C = %s" % channel_capacity)
    plt.show()


# Example usage (assuming you have a Q-table named q_table):
# visualize_q_table(q_table)

# Example for visualising the receiver's policy given a q-table with random values
# q_table = np.random.randint(0, 20, (25, 3, 4))
# print(str(q_table))
# visualize_receiver_policy(q_table, 25, 3, 2, [(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3), (2, 4), (4, 1), (4, 2), (4, 3)])

# Example for visualising the belief table given a belief table of 25 states and 3 messages
# belief_table = np.random.randint(0, 4, (25, 3))
# visualize_belief(belief_table, [(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3), (2, 4), (4, 1), (4, 2), (4, 3)])
