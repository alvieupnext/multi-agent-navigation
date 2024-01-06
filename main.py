import numpy as np

from environments.five_grid import FiveGrid, layouts as env_layouts
from agents.Sender import Sender
from agents.Receiver import Receiver
from agents.QLearningAgent import QLearningAgent
from plotting import visualize_belief, visualize_receiver_policy, visualize_thompson, plot_sorted_drop_with_confidence
from tqdm import tqdm

# Tqdm progress bar code is commented because it does not work with Ray jobs and makes the log files too big
# Can be uncommented when running locally
# # Flatten an array of messages into a single message (one hot encoding)
def flatten_messages(messages, num_messages):
    return sum([message * num_messages ** i for i, message in enumerate(messages)])

# Runs the Q-learning agent
def run_q_agent(gamma, epsilon_max, epsilon_min, epsilon_decay, learning_rate, env, learning_steps):
    options = {"termination_probability": 1 - gamma}
    receiver = QLearningAgent(env.world_size, gamma, learning_rate, epsilon_max, epsilon_min, epsilon_decay)
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
        # print("current_observation_and_mask: " + str(current_observation_and_mask))
        current_observation = current_observation_and_mask["observation"]
        # print("current_observation: " + str(current_observation))
        current_mask = current_observation_and_mask["action_mask"]
        # print("current_mask: " + str(current_mask))
        action = receiver.choose_action(current_observation, current_mask)
        # print("action: " + str(action))
        next_observations, rewards, terminations, truncations, infos = env.step({"receiver": action})
        reward = rewards["receiver"]
        # print("reward: " + str(reward))
        next_observation_and_mask = next_observations["receiver"]
        # print("next_observation_and_mask: " + str(next_observation_and_mask))
        next_observation = next_observation_and_mask["observation"]
        # print("next_observation: " + str(next_observation))
        receiver.learn(current_observation, action, reward, next_observation)

        if terminations["receiver"] or truncations["receiver"]:
            receiver.update_epsilon()
            episode_steps.append(env.timestep)
            episodes_rewards.append(reward)
            episode_total_steps.append(step + 1)
            observations, infos = env.reset(options=options)
            goal_state = env.returnGoal()
            receiver.set_goal(goal_state)
            # print("end of episode")
        else:
            observations = next_observations
        # Update the progress bar
        # progress_bar.update(1)
        # progress_bar.set_description(f"Q-Learning Step {step}/{learning_steps}")

    # Close the progress bar
    # progress_bar.close()

    return episodes_rewards, episode_steps, episode_total_steps

# Runs experiment on the sender and receiver agents
def run_experiment(M, num_messages, alpha, epsilon_max, epsilon_min, epsilon_decay, gamma, env, learning_steps, random=False):
    # C = num_messages ** M
    print(
        f"M: {M}, Number Of Possible Messages: {num_messages}, alpha: {alpha}, epsilon_max = {epsilon_max}, epsilon_min = {epsilon_min}, decay_rate = {epsilon_decay} gamma: {gamma}")
    options = {"termination_probability": 1 - gamma}
    senders = [Sender(num_messages, env.world_size, alpha) for _ in range(M)]
    receiver = Receiver(env.world_size, num_messages ** M, gamma, alpha, epsilon_max, epsilon_min, epsilon_decay)
    messages = []
    observations, infos = env.reset(options=options)
    goal_state = env.returnGoal()

    for sender in senders:
        if random:
            messages.append(sender.choose_action_random())
        else:
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
        receiver.learn(observation["observation"], message, action, reward, next_observation["observation"])

        if terminations["receiver"] or truncations["receiver"]:
            for sender, message in zip(senders, messages):
                sender.learn(goal_state, message, reward)
            receiver.update_epsilon()
            episode_steps.append(env.timestep)
            episodes_rewards.append(reward)
            episode_total_steps.append(step + 1)
            observations, infos = env.reset(options=options)
            goal_state = env.returnGoal()
            messages = []
            for sender in senders:
                if random:
                    messages.append(sender.choose_action_random())
                else:
                    messages.append(sender.choose_action(goal_state))
            message = flatten_messages(messages, num_messages)
        else:
            observations = next_observations

    # # Check if the sender and receiver have learned the correct policy
    # for sender in senders:
    #     print(f"Sender Q-Table: {sender.belief_table}")
    #     print(f"Receiver Q-Table: {receiver.q_table}")

    # Update the progress bar
    # progress_bar.update(1)
    # progress_bar.set_description(f"Step {step}/{learning_steps}, M: {M}, Number Of Possible Messages: {num_messages}")

    # Close the progress bar
    # progress_bar.close()
    # Visualize the belief table
    # for sender in senders:
    #     visualize_thompson(sender.alphas, sender.betas, num_messages, chosen_layout)
    # for message in range(num_messages ** M):
    #     visualize_receiver_policy(receiver.q_table, env.world_size, num_messages ** M, message, chosen_layout)
    return episodes_rewards, episode_steps, episode_total_steps, senders, receiver

    # Any additional logic or cleanup

# Experiment code for figure 6
def run_scrambled_experiments(senders, receiver, env, evaluation_episodes):
    all_results = []
    gamma = 0.8
    num_messages = senders[0].num_possible_messages
    options = {"termination_probability": 1 - gamma}
    # Make a sequence that goes from evaluation_steps to total_steps in steps of evaluation_steps
    # Set the epsilon of the receiver to 0
    receiver.epsilon = 0
    for i in range(6):
        scramble_index = None if i == 0 else i - 1
        messages = []
        episodes_rewards = []
        observations, infos = env.reset(options=options)
        goal_state = env.returnGoal()

        for sender in senders:
            messages.append(sender.choose_action(goal_state))

        # If the scramble index is not None, scramble the messages at the given index
        if scramble_index is not None:
            messages[scramble_index] = np.random.choice(num_messages)

        # Flatten the messages into a single message
        message = flatten_messages(messages, num_messages)
        episode_count = 0
        #Regular results, no scrambling
        while True:
            observation = observations["receiver"]
            action = receiver.choose_action(observation["observation"], message, observation["action_mask"])
            next_observations, rewards, terminations, truncations, infos = env.step({"receiver": action})
            reward = rewards["receiver"]

            if terminations["receiver"] or truncations["receiver"]:
                episodes_rewards.append(reward * gamma ** env.timestep)
                episode_count += 1
                if episode_count == evaluation_episodes:
                    break
                observations, infos = env.reset(options=options)
                goal_state = env.returnGoal()
                messages = []
                for sender in senders:
                    messages.append(sender.choose_action(goal_state))
                # If the scramble index is not None, scramble the messages at the given index
                if scramble_index is not None:
                    messages[scramble_index] = np.random.choice(num_messages)
                message = flatten_messages(messages, num_messages)
            else:
                observations = next_observations
        all_results.append(episodes_rewards)
    return all_results

# Function to perform bootstrapping
def bootstrap_confidence_interval(regular_results, scrambled_results, n_samples):
    bootstrap_samples = []
    for _ in range(n_samples):
        # Sample with replacement
        sampled_regular = np.random.choice(regular_results, size=len(regular_results), replace=True)
        sampled_scrambled = np.random.choice(scrambled_results, size=len(scrambled_results), replace=True)

        # Calculate drop
        drop = calculate_drop(sampled_regular, sampled_scrambled)
        bootstrap_samples.append(drop)

    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrap_samples, 2.5)
    upper_bound = np.percentile(bootstrap_samples, 97.5)
    return lower_bound, upper_bound

# Function to calculate drop
def calculate_drop(regular_results, scrambled_results):
    regular_mean = np.mean(regular_results)
    scrambled_mean = np.mean(scrambled_results)
    return (regular_mean - scrambled_mean) / regular_mean * 100


# Experiment to generate Figure 6 in the paper
# # Generate a plot for the rewards and steps
# # 4 million steps
# learning_steps = 4_000_000
# gamma = 0.8
# # We evaluate over 1000 episodes
# evaluation_episodes = 1000
# # Figure 6 in the paper
# regular_results, scrambled0_results, scrambled1_results,\
#         scrambled2_results, scrambled3_results, scrambled4_results = [], [], [], [], [], []
# # Define senders and receiver here
# for lay_out in env_layouts.values():
#     env = FiveGrid(illegal_positions=lay_out)
#     # Train the senders and receiver here
#     _, _, _, senders, receiver = run_experiment(5, 2, 0.001, 1, 0.01, 0.9999951365, gamma, env, learning_steps)
#     regular_result, scrambled0_result, scrambled1_result,\
#         scrambled2_result, scrambled3_result, scrambled4_result = run_scrambled_experiments(senders, receiver, env,
#                                                                                                evaluation_episodes)
#     regular_results.extend(regular_result)
#     scrambled0_results.extend(scrambled0_result)
#     scrambled1_results.extend(scrambled1_result)
#     scrambled2_results.extend(scrambled2_result)
#     scrambled3_results.extend(scrambled3_result)
#     scrambled4_results.extend(scrambled4_result)
#
# # Number of bootstrap samples
# n_bootstrap_samples = 1000
#
# mean_drops = []
# lower_bounds = []
# upper_bounds = []
# for scrambled_results in [scrambled0_results, scrambled1_results, scrambled2_results, scrambled3_results,
#                           scrambled4_results]:
#     mean_drop = calculate_drop(regular_results, scrambled_results)
#     mean_drops.append(mean_drop)
#     lower_bound, upper_bound = bootstrap_confidence_interval(regular_results, scrambled_results, n_bootstrap_samples)
#     lower_bounds.append(lower_bound)
#     upper_bounds.append(upper_bound)
#
#
# plot_sorted_drop_with_confidence(mean_drops, list(zip(lower_bounds, upper_bounds)))