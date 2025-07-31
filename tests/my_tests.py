import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

def test_adaptive_management_base():

    transition_function = np.array([
                            1, 0, 0, 1,
                            0.9, 0.1, 0, 1,
                            1, 0, 0, 1,
                            1, 0, 0, 1
                        ]).reshape(2, 2, 2, 2)

    reward_function = np.array([
                      0.736, 0.735,
                      0.736, 0.8540772
                        ]).reshape(2, 2)

    env = gym.make('AdaptiveManagement-v0',
                    params = {"init_state": 0,
                            "transition_function": transition_function,
                            "reward_function": reward_function},
                    Tmax=100)
    # If the environment doesn't follow the interface, an error will be thrown
    check_env(env, warn=True)

def test_adaptive_management_development():

    env = gym.make('TechnoDevEnv-v0')
    # If the environment doesn't follow the interface, an error will be thrown
    check_env(env, warn=True)
