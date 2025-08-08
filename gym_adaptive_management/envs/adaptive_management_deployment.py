import gymnasium as gym
import numpy as np
import pandas as pd
from gym_adaptive_management.common.read_IPCC_data import load_IPCC_dataset


class TechDeployEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an environment for technology deployment as an adaptive management problems
    with a coral reef following a logistic growth
    params:
        values_delta_t_K : list of competing values for delta_t_K
        scenario : climate change scenario (string)
        init_belief : prior distribution on the values of values_delta_t_K
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    def __init__(
                self,
                params = {"values_delta_t_K":np.array([1,2.5]),
                        "scenario": "SSP2_4_5",
                        "init_belief": np.array([0.5,0.5])},
                Tmax=100,
                see_belief=True,
                render_mode="console",
    ):
            super(TechDeployEnv, self).__init__()

            self.render_mode = render_mode
            self.see_belief = see_belief

            df = load_IPCC_dataset()
            self.scenario = params["scenario"]
            self.data_scenario = df[df['scenario'] == self.scenario]
            self.data_temperatures = df['Mean'] #average temperature
            self.Tmax = len(self.data_temperatures)

            # Parameters of the model
            self.init_state = 0.8 #initial state
            self.state = self.init_state
            self.time_step = 0

            self.r = 0.2 #growth rate
            self.Kmin = 0
            self.Kmax = 1
            self.m = 5 # rate of habitat loss

            #parameters MDP
            self.N_actions = 3 #0, 1, 2
            self.N_states = 11 # 0 to 1 by 0.1
            self.vect_of_states = np.arange(self.N_states)/10
            self.values_delta_t_K = params["values_delta_t_K"]
            self.N_models = len(self.values_delta_t_K)


            #if the initial belief is not provided, uniform belief is the default.
            if "init_belief" in params:
                self.init_belief = params["init_belief"]
                self.belief = params["init_belief"]
            else:
                self.init_belief = np.ones(self.N_models)/self.N_models
                self.belief = np.ones(self.N_models)/self.N_models

            #if the true model index is not provided, the model is sampled according to the initial belief
            if "true_delta_K" in params:
                self.true_delta_K = params["true_delta_K"]
                self.random_model = False
            else:
                self.true_delta_K = np.random.choice(self.values_delta_t_K, 1, p=self.init_belief)[0]
                self.random_model = True

            if "discount_factor" in params:
                self.discount_factor = params["discount_factor"]
            else:
                self.discount_factor = 0.9

            # Define action and observation space
            # They must be gym.spaces objects
            # Example when using discrete actions, we have two: left and right
            # The observation will be the coordinate of the agent
            # this can be described both by Discrete and Box space
            self.action_space = spaces.Discrete(self.N_actions)

            if self.see_belief:
                self.observation_space = spaces.Dict({
                    "state": spaces.Discrete(self.N_states),
                    "belief": spaces.Box(0, 1, shape=(self.N_models,))
                })
            else:
                self.observation_space = spaces.Dict({
                    "state": spaces.Discrete(self.N_states)
                })


    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed, options=options)

        self.state = self.init_state
        self.belief = self.init_belief
        self.time_step = 0

        #if the true model index is not provided, the model is sampled according to the initial belief
        if self.random_model:
            self.true_model_index = np.random.choice([0,1], 1, p=self.init_belief)[0]
            self.true_transition_model = self.transition_function[self.true_model_index]

        state = int(self.state)  # 0 or 1
        belief = self.belief.copy().astype(np.float32)
        #time_step = int(self.time_step)

        if self.see_belief:
            observation = {"state": state, "belief": belief}
        else:
            observation = {"state": state}

        info = {}

        return observation, info

    def transition_logistic(self, action, current_state):
        """
        transition function as a logistic function with noise as a log-normal distribution

        """
        if self.K>0.05:#for numerical purposes, to avoid dividing by 0
            state_bar = current_state + self.r*current_state(1-current_state/self.K)

            next_state = np.exp(np.random.normal(np.log(state_bar), self.sigma))
        else:
            next_state = 0
        return(next_state)

    def K_value(self,action):
        """
        calculate the carrying capacity at each time step
        """
        current_temp = self.data_temperatures[self.time_step]
        self.K = self.Kmin + (self.Kmax - self.Kmin)(1-1/(1+np.exp(self.m*(current_temp-action-self.true_delta_K))))
