import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import numpy as np
import gym_adaptive_management
from gym_adaptive_management.envs.adaptive_management_development import TechnoDevEnv
from stable_baselines3.common.monitor import Monitor

class FlattenAndOneHotEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        base_env = env.unwrapped  # get raw env with N_states, N_models

        self.N_states = base_env.N_states
        self.N_models = base_env.N_models

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(self.N_states + self.N_models,),
            dtype=np.float32
        )

    def observation(self, obs):
        state_oh = np.zeros(self.N_states, dtype=np.float32)
        state_oh[obs["state"]] = 1.0
        belief = obs["belief"].astype(np.float32)
        return np.concatenate([state_oh, belief])


class FlattenOneHotNoBeliefEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        base_env = env.unwrapped  # get raw env with N_states, N_models

        self.N_states = base_env.N_states
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(self.N_states,), dtype=np.float32
        )

    def observation(self, obs):
        state_oh = np.zeros(self.N_states, dtype=np.float32)
        state_oh[obs["state"]] = 1.0
        return state_oh


class FlattenOneHotNoStateEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        base_env = env.unwrapped  # get raw env with N_states, N_models

        self.N_models = base_env.N_models
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(self.N_models,), dtype=np.float32
        )

    def observation(self, obs):
        belief = obs["belief"].astype(np.float32)
        return belief
