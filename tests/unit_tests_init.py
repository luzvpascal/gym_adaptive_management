import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from gymnasium.envs.registration import register
from gym_adaptive_management.envs.adaptive_management_base import AdaptiveManagement
from gym_adaptive_management.envs.adaptive_management_development import TechnoDevEnv


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

env = AdaptiveManagement(
            params = {"init_state": 0,
                      "transition_function": transition_function,
                      "reward_function": reward_function},
            Tmax=100)

check_env(env, warn=True)
print("Environment successfully created")

env2 = gym.make('AdaptiveManagement-v0',
                params = {"init_state": 0,
                        "transition_function": transition_function,
                        "reward_function": reward_function},
                Tmax=100)
# If the environment doesn't follow the interface, an error will be thrown
check_env(env2, warn=True)
print("Environment 2 successfully created")

# env3 = TechnoDevEnv()
env3 = gym.make('TechnoDevEnv-v0')
# If the environment doesn't follow the interface, an error will be thrown
check_env(env, warn=True)
print("Environment 3 successfully created")

print("AdaptiveManagement-v0 in registry:")
print('AdaptiveManagement-v0' in gym.envs.registry.keys())

print("TechnoDevEnv-v0 in registry:")
print('TechnoDevEnv-v0' in gym.envs.registry.keys())
