import functools
from copy import copy

from gymnasium.spaces import Discrete
from numpy import int8
from pettingzoo import ParallelEnv
import numpy as np
from pettingzoo.utils import wrappers, parallel_to_aec

