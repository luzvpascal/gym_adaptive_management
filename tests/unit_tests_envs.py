import gymnasium as gym
import gym_adaptive_management
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
#
print("AdaptiveManagement-v0 in registry:")
print('AdaptiveManagement-v0' in gym.envs.registry.keys())

print("TechnoDevEnv-v0 in registry:")
print('TechnoDevEnv-v0' in gym.envs.registry.keys())


print("TechnoDevEnvOneHot-v0 in registry:")
print('TechnoDevEnvOneHot-v0' in gym.envs.registry.keys())

print("TechnoDevEnvNoBelief-v0 in registry:")
print('TechnoDevEnvNoBelief-v0' in gym.envs.registry.keys())


print("TechnoDevEnvNoState-v0 in registry:")
print('TechnoDevEnvNoState-v0' in gym.envs.registry.keys())

# transition_function = np.array([
#                         1, 0, 0, 1,
#                         0.9, 0.1, 0, 1,
#                         1, 0, 0, 1,
#                         1, 0, 0, 1
#                     ]).reshape(2, 2, 2, 2)
#
# reward_function = np.array([
#                   0.736, 0.735,
#                   0.736, 0.8540772
#                     ]).reshape(2, 2)

# env = AdaptiveManagement(
#             params = {"init_state": 0,
#                       "transition_function": transition_function,
#                       "reward_function": reward_function},
#             Tmax=100)
#
# check_env(env, warn=True)
# print("Environment successfully created")

# env2 = gym.make('AdaptiveManagement-v0',
#                 params = {"init_state": 0,
#                         "transition_function": transition_function,
#                         "reward_function": reward_function},
#                 Tmax=100)
# # If the environment doesn't follow the interface, an error will be thrown
# check_env(env2, warn=True)
# print("Environment 2 successfully created")

env = gym.make('TechnoDevEnv-v0')
check_env(env, warn=True)
print("TechnoDevEnv successfully created")

env = gym.make('TechnoDevEnvOneHot-v0')
check_env(env, warn=True)
print("TechnoDevEnvOneHot successfully created")


env = gym.make('TechnoDevEnvNoBelief-v0')
check_env(env, warn=True)
print("TechnoDevEnvNoBelief successfully created")


env = gym.make('TechnoDevEnvNoState-v0')
check_env(env, warn=True)
print("TechnoDevEnvNoState successfully created")

## train
print("Training started TechnoDevEnvOneHot")
vec_env = make_vec_env("TechnoDevEnvOneHot", n_envs=2)
#model = A2C("MultiInputPolicy", vec_env, verbose=1).learn(5000)
# model =  PPO("MultiInputPolicy", vec_env, verbose=2, gamma=0.9).learn(100000)
model =  DQN("MlpPolicy", vec_env, verbose=1, gamma=0.8).learn(1000)
print("Training completed")

print("Training started TechnoDevEnvNoBelief")
vec_env = make_vec_env("TechnoDevEnvNoBelief", n_envs=1)
#model = A2C("MultiInputPolicy", vec_env, verbose=1).learn(5000)
# model =  PPO("MultiInputPolicy", vec_env, verbose=2, gamma=0.9).learn(100000)
model =  DQN("MlpPolicy", vec_env, verbose=1, gamma=0.8).learn(1000)
print("Training completed")

print("Training started TechnoDevEnvNoState")
vec_env = make_vec_env("TechnoDevEnvNoState", n_envs=1)
#model = A2C("MultiInputPolicy", vec_env, verbose=1).learn(5000)
# model =  PPO("MultiInputPolicy", vec_env, verbose=2, gamma=0.9).learn(100000)
model =  DQN("MlpPolicy", vec_env, verbose=1, gamma=0.8).learn(1000)
print("Training completed")
