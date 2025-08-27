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

################################################################################
################################################################################
## Development ##################################################################
################################################################################
################################################################################
print("TechnoDevEnv-v0 in registry:")
print('TechnoDevEnv-v0' in gym.envs.registry.keys())

print("TechnoDevEnvOneHot-v0 in registry:")
print('TechnoDevEnvOneHot-v0' in gym.envs.registry.keys())

print("TechnoDevEnvNoBelief-v0 in registry:")
print('TechnoDevEnvNoBelief-v0' in gym.envs.registry.keys())


print("TechnoDevEnvNoState-v0 in registry:")
print('TechnoDevEnvNoState-v0' in gym.envs.registry.keys())

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
model =  DQN("MlpPolicy", vec_env, verbose=0, gamma=0.8).learn(1000)
print("Training completed")

print("Training started TechnoDevEnvNoBelief")
vec_env = make_vec_env("TechnoDevEnvNoBelief", n_envs=2)
model =  DQN("MlpPolicy", vec_env, verbose=0, gamma=0.8).learn(1000)
print("Training completed")

print("Training started TechnoDevEnvNoState")
vec_env = make_vec_env("TechnoDevEnvNoState", n_envs=2)
model =  DQN("MlpPolicy", vec_env, verbose=0, gamma=0.8).learn(1000)
print("Training completed")


################################################################################
################################################################################
## Deployment ##################################################################
################################################################################
################################################################################

print("TechnoDeployEnv-v0 in registry:")
print('TechnoDeployEnv-v0' in gym.envs.registry.keys())

print("TechnoDeployEnvOneHot-v0 in registry:")
print('TechnoDeployEnvOneHot-v0' in gym.envs.registry.keys())

print("TechnoDeployEnvNoBelief-v0 in registry:")
print('TechnoDeployEnvNoBelief-v0' in gym.envs.registry.keys())


print("TechnoDeployEnvNoState-v0 in registry:")
print('TechnoDeployEnvNoState-v0' in gym.envs.registry.keys())


env = gym.make('TechnoDeployEnv-v0')
check_env(env, warn=True)
print("TechnoDeployEnv successfully created")

env = gym.make('TechnoDeployEnvOneHot-v0')
check_env(env, warn=True)
print("TechnoDeployEnvOneHot successfully created")


env = gym.make('TechnoDeployEnvNoBelief-v0')
check_env(env, warn=True)
print("TechnoDeployEnvNoBelief successfully created")


env = gym.make('TechnoDeployEnvNoState-v0')
check_env(env, warn=True)
print("TechnoDeployEnvNoState successfully created")

## train
print("Training started TechnoDeployEnvOneHot")
vec_env = make_vec_env("TechnoDeployEnvOneHot", n_envs=2)
model =  DQN("MlpPolicy", vec_env, verbose=0, gamma=0.8).learn(1000)
print("Training completed")

print("Training started TechnoDeployEnvNoBelief")
vec_env = make_vec_env("TechnoDeployEnvNoBelief", n_envs=2)
model =  DQN("MlpPolicy", vec_env, verbose=0, gamma=0.8).learn(1000)
print("Training completed")

print("Training started TechnoDeployEnvNoState")
vec_env = make_vec_env("TechnoDevEnvNoState", n_envs=2)
model =  DQN("MlpPolicy", vec_env, verbose=0, gamma=0.8).learn(1000)
print("Training completed")
