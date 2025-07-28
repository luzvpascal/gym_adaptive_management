import gymnasium as gym
from gym.envs.registration import register
from gym_adaptive_management.envs.adaptive_management_base import AdaptiveManagement

register(
    id='AdaptiveManagement-v0',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_base:AdaptiveManagement',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)
