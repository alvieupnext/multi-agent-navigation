import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def plot_reward_curves(pdDataframe):
    """
    For each of the environments, plot the reward curves for the different settings.
    """
    # environments = ["pong", "four_room", "two_room", "flower", "empty_room"]
    for env in environments:
        plot_reward_curve(pdDataframe[pdDataframe["Env"] == env], env)


environments = ["pong", "four_room", "two_room", "flower", "empty_room"]

gamma = 0.8


def plot_reward_curve(pdDataframe, environment, gamma=0.99):
    settings = ["5S-1R", "4S-1R", "3S-1R", "2S-1R", "1S-1R", "Random", "Q-learning", "Max"]
    fig, ax = plt.subplots()

    for setting in settings:
        data = pdDataframe[pdDataframe["Type"] == setting]

        # Group by TotalSteps and get for each group the mean and standard deviation of the DiscountedReward
        grouped_data = data.groupby("TotalSteps")["DiscountedReward"].agg(["mean", "std"]).reset_index()

        # Calculate the rolling mean and standard deviation
        rolling_mean = grouped_data["mean"].rolling(window=1000).mean()
        rolling_std = grouped_data["std"].rolling(window=1000).mean()
        # Plot mean and error zone
        ax.plot(grouped_data['TotalSteps'], rolling_mean, label=setting)
        ax.fill_between(grouped_data['TotalSteps'], rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.5)

    ax.set_title(f"Reward curves with error zones for {environment}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.legend()
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

        ax.set_title(environment)
        ax.set_xlabel("Channel Capacity")

        # Set the x ticks to be equally spaced and label them with the channel capacities
        ax.set_xticks(grouped_data.index)
        ax.set_xticklabels(grouped_data['C'])

        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Average Return")

    plt.tight_layout()
    plt.show()


# print("before reading")
# Open tabular_results_pong_0.0001_0.05_0.05_0.8_12000000 from results folder
# df = pd.read_csv("results/tabular_results_pong_0.0001_0.05_0.05_0.8_12000000.csv")
# print("after reading")

# Plot the reward curves
# plot_reward_curve(df, "pong")


# # Load tabular_results_pong_0.0001_0.05_0.05_0.8_12000000.csv from the results folder as a pandas dataframe
# df = pd.read_csv("results/tabular_results_pong_0.0001_0.05_0.05_0.8_12000000.csv")
# # Plot the reward curves
# plot_reward_curve(df, "pong")


#
# Generate mock data for testing the function
#

# Generate mock data for testing the function
np.random.seed(0)  # For reproducibility

# Define parameters for mock data
num_episodes = 10_000  # Number of episodes
environments = ["pong", "four_room", "two_room", "flower", "empty_room"]
settings = ['5S-1R']
channel_capacity = [3, 4, 5, 8, 9, 16, 25, 27, 32, 64]

# Create an empty DataFrame
df = pd.DataFrame()

# Populate the DataFrame
for env in environments:
    for setting in settings:
        for c in channel_capacity:
            # Generate a linearly increasing reward with some noise
            reward = np.linspace(0, 1, num_episodes) + np.random.normal(0, 0.1, num_episodes)
            reward = np.clip(reward, 0, 1)  # Ensure rewards are between 0 and 1

            # Generate other columns
            episodes = np.arange(0, num_episodes)
            steps = np.random.randint(1, 100, num_episodes)
            total_steps = np.cumsum(steps)

            # Create a temp DataFrame and append to the main DataFrame
            temp_df = pd.DataFrame({
                'Episodes': episodes,
                'Reward': reward,
                'Steps': steps,
                'TotalSteps': total_steps,
                'C': c,
                'Type': setting,
                'Env': env
            })
            df = pd.concat([df, temp_df])

# Reset index
df = df.reset_index(drop=True)

# print(df)

# Calculate the discounted rewards and add them to the DataFrame
df['DiscountedReward'] = df['Reward'] * gamma ** df['Steps']


# Plot the reward curves
# plot_reward_curves(df)

# plot_average_return_per_channel_capacity_excluding(df, environments, ["Random", "Q-learning", "Max"])

def visualize_belief(belief_table, layout):
    indices = np.argmax(belief_table, axis=1)
    # get max index
    wall = np.max(indices) + 1
    indices = indices.reshape((5, 5))
    print(indices)
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
