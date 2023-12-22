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

observations, infos = env.reset()

while env.agents:
    # From observations, get the action mask
    action_mask = observations["receiver"]["action_mask"]
    # Get the observations, rewards, terminations, truncations and infos
    observations, rewards, terminations, truncations, infos = env.step({"receiver": choose_action(action_mask)})
    # Print receiver reward
    print("Receiver reward: " + str(rewards["receiver"]))
    # Render the environment
    env.render()

