from environments.five_grid import FiveGrid, layouts
import numpy as np

chosen_layout = layouts["pong"]
env = FiveGrid(illegal_positions=chosen_layout)

#Generate a generic loop for the environment using an agent that does random actions
# Path: main.py
# Compare this snippet from agents/Receiver.py:
def choose_action(action_mask):
    # Choose a random action
    action = np.random.choice(np.where(action_mask == 1)[0])
    # Return the action
    return action

# Termination probability of the environment
options = {"termination_probability": 0.1}

observations, infos = env.reset(options=options)

env.render()

while env.agents:
    # From observations, get the action mask
    action_mask = observations["receiver"]["action_mask"]
    choice = choose_action(action_mask)
    # Get the observations, rewards, terminations, truncations and infos
    observations, rewards, terminations, truncations, infos = env.step({"receiver": choice})
    # Print receiver reward
    print("Receiver reward: " + str(rewards["receiver"]))
    # Render the environment
    env.render()

