import gymnasium as gym
import gym_adaptive_management
import numpy as np
import pandas as pd
from gym_adaptive_management.common.read_IPCC_data import load_IPCC_dataset

df = load_IPCC_dataset()
print(df)
