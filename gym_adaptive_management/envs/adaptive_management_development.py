import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_adaptive_management.envs.adaptive_management_base import AdaptiveManagement


class TechnoDevEnv(AdaptiveManagement):
    """
    This class defines the problem of technology development with uncertain development success
    The POMDP is defined as:
    X = {idle, ready}, the states of the technology
    Y = {success, failure} the hidden models of technology development
    A = {do nothing, invest in R&D/deploy} the action space
    T = [1, 0] if y = failure, for both actions, or if y=success and action=do nothing
        [0, 1]

        [1-pdev, pdev] if y = success and action = invest in R&D/deploy
        [0, 1]
              do nothing  invest in R&D/deploy
    R = idle [Rbau,         Rbau - Cdev]
        ready[Rbau,         Rdep]

    INPUTS:
        pdev: probability of the technology transitionning from idle to ready if y = success and action = invest in R&D/deploy
        Rbau: baseline rewards when action = do nothing
        Rdep: expect rewards if the technology is successfully developed
        Cdev: development costs
        init_belief_success: initial belief in development success
    """
    def __init__(
        self,
        p_dev = 0.1,
        Rbau = 0,
        Rdep = 0.1,
        Cdev = 0.001,
        randomize_initial_belief = True,
        init_belief_success = 0.5,
        Tmax=100,
        discount_factor=0.95
    ):
        transition_function = np.array([
                                1, 0, 0, 1,
                                1-p_dev, p_dev, 0, 1,
                                1, 0, 0, 1,
                                1, 0, 0, 1
                            ]).reshape(2, 2, 2, 2)

        reward_function = np.array([
                          Rbau, Rbau - Cdev,
                          Rbau, Rdep
                            ]).reshape(2, 2)

        super().__init__(
            params={"init_state": 0,
                    "transition_function": transition_function,
                    "reward_function": reward_function,
                    "init_belief": np.array([init_belief_success, 1-init_belief_success]),
                    "randomize_initial_belief": randomize_initial_belief,
                    "discount_factor": discount_factor},
            Tmax=Tmax,
            see_belief=True,
            render_mode="console",
        )

class TechnoDevEnvNoBelief(AdaptiveManagement):
    """
    This class defines the problem of technology development with uncertain development success
    The POMDP is defined as:
    X = {idle, ready}, the states of the technology
    Y = {success, failure} the hidden models of technology development
    A = {do nothing, invest in R&D/deploy} the action space
    T = [1, 0] if y = failure, for both actions, or if y=success and action=do nothing
        [0, 1]

        [1-pdev, pdev] if y = success and action = invest in R&D/deploy
        [0, 1]
              do nothing  invest in R&D/deploy
    R = idle [Rbau,         Rbau - Cdev]
        ready[Rbau,         Rdep]

    INPUTS:
        pdev: probability of the technology transitionning from idle to ready if y = success and action = invest in R&D/deploy
        Rbau: baseline rewards when action = do nothing
        Rdep: expect rewards if the technology is successfully developed
        Cdev: development costs
        init_belief_success: initial belief in development success
    In this problem, the agent does not see the evolving belief
    """
    def __init__(
        self,
        p_dev = 0.1,
        Rbau = 0.7,
        Rdep = 0.8,
        Cdev = 0.001,
        init_belief_success = 0.5,
        Tmax=100,
    ):
        transition_function = np.array([
                                1, 0, 0, 1,
                                1-p_dev, p_dev, 0, 1,
                                1, 0, 0, 1,
                                1, 0, 0, 1
                            ]).reshape(2, 2, 2, 2)

        reward_function = np.array([
                          Rbau, Rbau - Cdev,
                          Rbau, Rdep
                            ]).reshape(2, 2)

        super().__init__(
            params={"init_state": 0,
                    "transition_function": transition_function,
                    "reward_function": reward_function,
                    "init_belief": np.array([init_belief_success, 1-init_belief_success])},
            Tmax=Tmax,
            see_belief=False,
            render_mode="console",
        )
