import gymnasium as gym
from gymnasium.envs.registration import register
from gym_adaptive_management.envs.adaptive_management_base import AdaptiveManagement
from gym_adaptive_management.envs.adaptive_management_beliefMDP import AdaptiveManagementBeliefMDP

from gym_adaptive_management.envs.adaptive_management_development import TechnoDevEnv, TechnoDevEnvNoBelief
from gym_adaptive_management.envs.adaptive_management_development_beliefMDP import TechnoDevEnvBeliefMDP, TechnoDevEnvNoBeliefBeliefMDP

#adaptive management general
register(
    id='AdaptiveManagement-v0',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_base:AdaptiveManagement',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)
register(
    id='AdaptiveManagementBeliefMDP-v0',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_beliefMDP:AdaptiveManagementBeliefMDP',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)

#adaptive management technology development
register(
    id='TechnoDevEnv-v0',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_development:TechnoDevEnv',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)

register(
    id='TechnoDevEnv-v1',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_development:TechnoDevEnvNoBelief',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)

register(
    id='TechnoDevEnvBeliefMDP-v0',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_development_beliefMDP:TechnoDevEnvBeliefMDP',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)

register(
    id='TechnoDevEnvBeliefMDP-v1',  # Environment ID, used to make the environment
    entry_point='gym_adaptive_management.envs.adaptive_management_development_beliefMDP:TechnoDevEnvNoBeliefBeliefMDP',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)
