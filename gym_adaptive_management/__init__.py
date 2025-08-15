import gymnasium as gym
from gymnasium.envs.registration import register
from gym_adaptive_management.envs.adaptive_management_base import AdaptiveManagement
from gym_adaptive_management.envs.adaptive_management_development import TechnoDevEnv

from gym_adaptive_management.envs.wrappers import FlattenAndOneHotEnv, FlattenOneHotNoBeliefEnv
from stable_baselines3.common.monitor import Monitor
from typing import Callable, Optional

##################################
# Register coded environments ####
##################################

#adaptive management general
register(
    id='AdaptiveManagement-v0',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_base:AdaptiveManagement',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)

#adaptive management technology development
register(
    id='TechnoDevEnv-v0',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_development:TechnoDevEnv',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)

registered_envs = ['TechnoDevEnv-v0']#add new environments every time you create one

##################################
# Register FlattenAndOneHotEnv####
##################################
def create_FlattenAndOneHotEnv(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_FlattenAndOneHotEnv() -> gym.Env:
        env = gym.make(env_id)
        env = FlattenAndOneHotEnv(env)
        env = Monitor(env)
        return env
    return make_FlattenAndOneHotEnv

for env_id in registered_envs:
    name, version = env_id.split("-v")
    register(
        id=f"{name}OneHot-v{version}",
        entry_point=create_FlattenAndOneHotEnv(env_id),  # type: ignore[arg-type]
    )

#####################################
# Register FlattenOneHotNoBeliefEnv##
#####################################
def create_FlattenOneHotNoBeliefEnv(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_FlattenOneHotNoBeliefEnv() -> gym.Env:
        env = gym.make(env_id)
        env = FlattenOneHotNoBeliefEnv(env)
        env = Monitor(env)
        return env
    return make_FlattenOneHotNoBeliefEnv

for env_id in registered_envs:
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoBelief-v{version}",
        entry_point=create_FlattenOneHotNoBeliefEnv(env_id),  # type: ignore[arg-type]
    )
