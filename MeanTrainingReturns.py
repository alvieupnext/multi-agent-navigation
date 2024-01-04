import ray
import pandas as pd
from main import run_experiment, FiveGrid, env_layouts, run_q_agent

# The columns for the dataframe Episode, Reward, Steps, TotalSteps, C, type, env
#  type = 5S-1R, 4S-1R, 3S-1R, 2S-1R, 1S-1R, Q-learning, Random  S = Sender
# env = pong, four_room, two_room, flower, empty_room
columns = ["Episode", "Reward", "Steps", "TotalSteps", "C", "Type", "Env"]


@ray.remote
def run_training(type, C, eta, epsilon_max, epsilon_min, epsilon_decay, gamma, layout, learning_steps):
  # Create a new pandas dataframe to store the results
  # Get the correct lay-out
  env = FiveGrid(illegal_positions=env_layouts[layout])
  if type == "Q-learning":
    rewards, steps, total_steps = run_q_agent(gamma, epsilon_max, epsilon_min, epsilon_decay, eta,env, learning_steps)
  elif type == "Random":
    # Use a single sender with epsilon = 1
    rewards, steps, total_steps = run_experiment(1, C, eta, epsilon_max, epsilon_min,
                                                 epsilon_decay, gamma, env, learning_steps)
  else: # Type with senders and receivers
    # Types are of the kind 5S-1R, 4S-1R, 3S-1R, 2S-1R, 1S-1R
    # Get the first character of the type and convert it to an integer
    M = int(type[0])
    num_messages = int(C ** (1 / M))
    # Run the experiment
    rewards, steps, total_steps = run_experiment(M, num_messages, eta, epsilon_max, epsilon_min, epsilon_decay, gamma, env, learning_steps)
  # Generate a type from the M value
  # Create a dataframe from the results
  df = pd.DataFrame(list(zip(rewards, steps, total_steps)), columns=["Reward", "Steps", "TotalSteps"])
  # Add the episode number at the start of the dataframe
  df.insert(0, "Episode", df.index)
  # Add the C value to the dataframe
  df["C"] = C
  # Add the type to the dataframe
  df["Type"] = type
  # Add the environment to the dataframe
  df["Env"] = layout
  return df

# Eta is 1e-4
eta = 1e-4
# Gamma is 0.8
gamma = 0.8
# Epsilon values for the receiver
epsilon_max = 1
epsilon_min = 0.001
epsilon_decay = 0.9999951365
# C (excluding 27,32,36,64)
C = [3,4,5,8,9,16,25,27,32,36,64]
# Possible combintions
M = 5

possible_C_values = {
    1: [3, 4, 5, 8, 9, 16, 25, 27, 32, 36, 64],
    2: [4, 9, 16, 25, 36, 64],
    3: [8, 27, 64],
    4: [16],
    5: [32]
}
# Learning steps is 12 million
learning_steps = 4000000
# The possible layouts for the environment
layout = "four_room"
sender_receiver = ["5S-1R", "4S-1R", "3S-1R", "2S-1R", "1S-1R"]
# The possible types ("Q-learning" will be added soon)
types = ["5S-1R", "4S-1R", "3S-1R", "2S-1R", "1S-1R", "Random", "Q-learning"]



if __name__ == "__main__":
  ray.init(address='auto')
  # ray.init()
  remotes = []
  for type in types:
    # Get the first character of the type and convert it to an integer
      if type in sender_receiver or type == "Random":
        if type != "Random":
          M = int(type[0])
        else:
          M = 1
        # Get the possible C values for the type
        possible_C = possible_C_values[M]
        for C in possible_C:
          remotes.append(run_training.remote(type, C, eta, epsilon_max, epsilon_min, epsilon_decay, gamma, layout, learning_steps))
      else: #Q-learner
        remotes.append(run_training.remote(type, 1, eta, epsilon_max, epsilon_min, epsilon_decay, gamma, layout, learning_steps))
    # Create a new dataframe to store all the results
  df = pd.DataFrame(columns=columns)
  while len(remotes):
    done_remote, remotes = ray.wait(remotes, timeout=None)
    print("Finished a task")
    print("Number of tasks left: ", len(remotes))
    calculated_df = ray.get(done_remote[0])
    df = pd.concat([df, calculated_df], ignore_index=True)
  df.to_csv(f"tabular_results_{layout}_{eta}_{epsilon_max}_{epsilon_min}_{epsilon_decay}_{gamma}_{learning_steps}.csv")