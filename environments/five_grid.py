import functools
from copy import copy

from gymnasium.spaces import Discrete
from numpy import int8
from pettingzoo import ParallelEnv
import numpy as np
from pettingzoo.utils import wrappers, parallel_to_aec

# The first environment from the assignment
class FiveGrid(ParallelEnv):
    metadata = {
        "name": "five_grid",
        'render_modes': ['human'],
    }

    def __init__(self, illegal_positions=None, receiver_position=(2, 2), grid_size=(5, 5)):
        """The init methods takes in environment arguments.
        Should define the following attributes:
        - x and y coordinates for the receiver
        - x and y coordinates for the goal
        - grid size
        - the possible agents
        - a timestep counter
        """
        if illegal_positions is None:
            illegal_positions = []
        self.possible_agents = ["receiver"]
        self.receiver = None
        self.receiver_position = receiver_position
        self.grid_size = grid_size
        self.goal = ()
        self.illegal_positions = illegal_positions
        self.illegal_states = list(map(lambda x: self.flattenPosition(x[0], x[1]), self.illegal_positions))
        self.position_mapping = self.createStateMapping(self.grid_size[0] * self.grid_size[1], self.illegal_states)
        # Get all legal states by getting the values from the mapping (remove the Nones)
        self.mask_mapping = self.createMaskMapping(self.grid_size[0] * self.grid_size[1])
        self.num_states = (self.grid_size[0] * self.grid_size[1]) - len(self.illegal_states)
        self.timestep = 0
        self.ptem = 0

   # Generate an observation with their agent mask
    def generateObservation(self):
        #Get the flattened position of the agent
        agent_position = self.position_mapping[self.flattenPosition(self.receiver[0], self.receiver[1])]
        # Generate the observation for the agents
            # self.observation[agent] = self.agents[agent][0] + self.grid_size[1] * self.agents[agent][1]
        return {"observation": agent_position, "action_mask": self.mask_mapping[agent_position]}

    def flattenPosition(self, x, y):
        return x + self.grid_size[1] * y

    def unflattenPosition(self, position):
        return position % self.grid_size[1], position // self.grid_size[1]

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

               It needs to initialize the following attributes:
               - agents
               - timestamp
               - prisoner x and y coordinates
               - guard x and y coordinates
               - escape x and y coordinates
               - observation
               - infos

               And must set up the environment so that render(), step(), and observe() can be called without issues.
               """
        if options is None:
            options = {"termination_probability": 0.1}
        self.ptem = options["termination_probability"]
        self.agents = copy(self.possible_agents)
        self.receiver = self.receiver_position
        # For the goal location, we want to make sure that it is not in the same position as the receiver
        # We take it out of the legal positions stored in the position mapping keys but we filter out the keys
        # that have None as value
        # First, filter the mapping to remove None positions
        legal_positions = list(filter(lambda key: self.position_mapping[key] is not None, self.position_mapping.keys()))
        # flatten en translate the receiver position
        receiver_position = self.flattenPosition(self.receiver[0], self.receiver[1])
        # Remove the receiver position from the legal positions
        legal_positions.remove(receiver_position)
        # Choose a random position from the legal positions
        self.goal = self.unflattenPosition(np.random.choice(legal_positions))
        observations = {"receiver": self.generateObservation()}
        # Set the illegal actions (all the points that are not the doorway but on the doorway's column)
        # Move the action mask to the infos
        infos = {}
        return observations, infos


    def createStateMapping(self, num_states, illegal_states):
        """
        Create a mapping from original states to new states, excluding illegal states.

        :param num_states: The total number of states in the original state space.
        :param illegal_states: A list of states that are considered illegal and should be excluded.
        :return: A dictionary mapping from original states to new states.
        """
        new_state_mapping = {}
        new_state_index = 0

        for original_state in range(num_states):
            if original_state not in illegal_states:
                new_state_mapping[original_state] = new_state_index
                new_state_index += 1
            else:
                new_state_mapping[original_state] = None  # Mark illegal states with None

        return new_state_mapping

    def createMaskMapping(self, num_states):
        mask_mapping = {}
        # For every state, use the mapping to retreive the translated state
        for state in range(num_states):
            if self.position_mapping[state] is None:
                #Do Not generate a mask for illegal states
                continue
            else:
                translated_state = self.position_mapping[state]
                # Get the x and y coordinates of the translated state
                x, y = self.unflattenPosition(state)
                # Generate the action mask
                action_mask = np.ones(4, dtype=int8)
                # Out of bounds up
                if y == 0:
                    action_mask[0] = 0
                # Out of bounds down
                elif y == self.grid_size[1] - 1:
                    action_mask[1] = 0
                # Out of bounds left
                if x == 0:
                    action_mask[3] = 0
                # Out of bounds right
                elif x == self.grid_size[0] - 1:
                    action_mask[2] = 0
                # Generate possible next positions
                possible_next_positions = [(x, y - 1), (x, y + 1), (x + 1, y), (x - 1, y)]
                for i, possible_next_position in enumerate(possible_next_positions):
                    # If the possible next position is an illegal position, set the action mask to 0
                    if possible_next_position in self.illegal_positions:
                        action_mask[i] = 0
                mask_mapping[translated_state] = action_mask
        return mask_mapping



    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

                Needs to update:
                - receiver coordinates
                - timestamp
                - rewards
                - terminations
                - infos
                - truncations

                And any internal state used by observe() or render()
        """
        # Increment the timestep
        self.timestep += 1
        # Initialize the observations, rewards, terminations, truncations and infos
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        action = actions["receiver"]
        # Get the coordinates of the receiver
        x, y = self.receiver
        # Move the receiver in the direction of the action
        if action == 0 and y > 0:
            self.receiver = (x, y - 1)
        elif action == 1 and y < self.grid_size[1] - 1:
            self.receiver = (x, y + 1)
        elif action == 2 and x > 0:
            self.receiver = (x + 1, y)
        elif action == 3 and x < self.grid_size[0] - 1:
            self.receiver = (x - 1, y)
        # Check whether we have entered an illegal state
        if self.receiver in self.illegal_positions:
            print("Receiver is in an illegal position")
        # Update the observations
        observations["receiver"] = self.generateObservation()
        # Update the rewards
        rewards["receiver"] = 0
        # Update the terminations
        terminations["receiver"] = False
        # Update the truncations
        truncations["receiver"] = False
        # Update the infos
        infos["receiver"] = {}
        # Check whether the receiver has reached the goal
        if self.receiver == self.goal:
            terminations["receiver"] = True
            rewards["receiver"] = 1
        else:
            # Check for termination using the probability
            if np.random.random() < self.ptem:
                terminations["receiver"] = True

        # From the terminations, remove all the agents that have reached the goal
        for agent in terminations:
            if terminations[agent]:
                self.agents.remove(agent)

        #Return the observations, rewards, terminations, truncations and infos
        return observations, rewards, terminations, truncations, infos

    def render(self):
        # Render the environment to the screen
        grid = np.full(self.grid_size, fill_value=' ', dtype='<U1')  # Initialize with empty spaces
        grid[self.receiver[1], self.receiver[0]] = "R"  # Doorway
        grid[self.goal[1], self.goal[0]] = "G"  # Goal
        # Mark illegal positions with "XX"
        for x, y in self.illegal_positions:
            grid[y, x] = "X"
        print(f"{grid}\n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(self.num_states)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

#Here, we define the possible lay-outs
pong = [(0,1), (0,2), (0,3), (2,0), (2,1), (2,3), (2,4), (4,1), (4,2), (4,3)]
four_room = [(0,2), (1,2), (3,2), (4,2), (2,0), (2,4)]
two_room = [(2,0), (2,1), (2,3), (2,4)]
flower = [(0,2), (2, 0), (4,2), (2,4)]

#Make a dictionary for the lay-outs
layouts = {"pong": pong, "four_room": four_room, "two_room": two_room, "flower": flower}


