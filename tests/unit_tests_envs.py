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

print("AdaptiveManagement-v1 in registry:")
print('AdaptiveManagement-v1' in gym.envs.registry.keys())

print("TechnoDevEnv-v0 in registry:")
print('TechnoDevEnv-v0' in gym.envs.registry.keys())

print("TechnoDevEnv-v1 in registry:")
print('TechnoDevEnv-v1' in gym.envs.registry.keys())

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

# env3 = TechnoDevEnv()
env3 = gym.make('TechnoDevEnv-v0')
# # If the environment doesn't follow the interface, an error will be thrown
# check_env(env, warn=True)
print("Environment 3 successfully created")


# ## train
# print("Training started")
# vec_env = make_vec_env(TechnoDevEnv, n_envs=4)
# #model = A2C("MultiInputPolicy", vec_env, verbose=1).learn(5000)
# # model =  PPO("MultiInputPolicy", vec_env, verbose=2, gamma=0.9).learn(100000)
# model =  DQN("MultiInputPolicy", vec_env, verbose=1, gamma=0.8).learn(1000)
# print("Training completed")
#
# ## test environment
# env = TechnoDevEnv()
# obs, _ = env.reset()
# env.true_model_index = 1 #failure
# env.true_transition_model = env.transition_function[env.true_model_index]
# n_steps = 100
# for step in range(n_steps):
#     action, _ = model.predict(obs, deterministic=True)
#     print(f"Step {step + 1}")
#     print("Action: ", action)
#     obs, reward, terminated, truncated, info = env.step(action)
#     print("obs=", obs, "reward=", reward, "done=", terminated)
