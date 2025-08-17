import gymnasium as gym
import numpy as np
import pandas as pd
from scipy.stats import norm

from gym_adaptive_management.common.read_IPCC_data import load_IPCC_dataset
from gym_adaptive_management.envs.adaptive_management_base import AdaptiveManagement

class TechDeployEnv(AdaptiveManagement):
    """
    Custom Environment that follows gym interface.
    This is an environment for technology deployment as an adaptive management problems
    with a coral reef following a logistic growth
    INPUTS:
        values_delta_t_K : np.array of competing values for delta_t_K. Default [1,2.5].
        scenario : climate change scenario (string). Default "SSP2_4_5"
        init_state : initial coral cover. Default 0.8.
        init_belief : prior distribution on the values of values_delta_t_K. Default [0.5,0.5]
        randomize_initial_belief: bool indicating whether we need to randomize the initial belief. Default False.
        discount_factor: discount factor (float). Default 1.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    def __init__(
                self,
                values_delta_t_K=np.array([1,2.5]),
                scenario="SSP2_4_5",
                init_state=0.8,
                init_belief=np.array([0.5,0.5]),
                randomize_initial_belief = False,
                discount_factor=1,
                N_states=11
    ):
            super(TechDeployEnv, self).__init__()

            self.render_mode = render_mode

            df = load_IPCC_dataset()
            self.scenario = scenario
            self.data_scenario = df[df['scenario'] == self.scenario]
            self.data_temperatures = df['Mean'] #average temperature
            self.Tmax = len(self.data_temperatures)

            # Parameters of the model
            self.r = 0.2 #growth rate
            self.Kmin = 0
            self.Kmax = 1
            self.m = 5 # rate of habitat loss
            self.sigma_eco = 0.2

            #parameters MDP
            self.N_ecosystem = N_states # 0 to 1 by 0.1
            self.ecosystem_states = np.arange(self.N_ecosystem)/(self.N_ecosystem-1)
            self.values_delta_t_K = values_delta_t_K
            self.N_values_delta_t_K = len(self.values_delta_t_K)

            temp_discretization = 100
            Temp_max = round(max(df['X95.']))
            self.temperature_states = np.arange(0,Temp_max+Temp_max/temp_discretization,
                                                Temp_max/temp_discretization)
            self.N_temperatures = len(self.temperature_states)

            self.N_actions_transition = 3
            self.DEP_EFFECT = np.arange(self.N_actions_transition)

            super().__init__(
                params={"init_state": init_state,
                        "transition_function": transition_function,
                        "reward_function": reward_function,
                        "init_belief": init_belief,
                        "randomize_initial_belief": randomize_initial_belief,
                        "discount_factor": discount_factor,
                        "Tmax":Tmax},
                render_mode="console",
            )

    def K_function(self,delta_t_crit,delta_t):
        """
        calculate the carrying capacity for a given delta_t_crit at temperature delta_t
        delta_t_crit: shifting temperature
        delta_t: current local temperature
        """
        return self.Kmin + (self.Kmax - self.Kmin)(1-1/(1+np.exp(self.m*(delta_t-delta_t_crit))))

    def ecosystem_dynamics(self,x_t,K):
        """
        calculate the carrying capacity for a given delta_t_crit at temperature delta_t
        delta_t_crit: shifting temperature
        delta_t: current local temperature
        """
        return x_t + (x_t*self.r*(1-x_t/K)))

    def sik_bar_function(self, delta_t_crit):
        sik_bar = np.zeros((self.N_ecosystem, self.N_temperatures, self.N_actions_transition))
        for index_action in range(self.N_actions_transition):
            for index_state in range(self.N_ecosystem):
                for index_temp in range(self.N_temperatures):
                    K_eff = self.K_function(delta_t_crit, self.temperature_states[index_temp]-self.DEP_EFFECT[index_action])
                    sik_bar[index_state,
                            index_temp,
                            index_action] max(0, self.ecosystem_dynamics(self.ecosystem_states[index_state],
                                                                K_eff))
        return sik_bar

    def transition_function_ecosystem(self, sik_bar):
        min_prob = 0.001
        transition_ecosystem = np.zeros((self.N_actions_transition,
                                        self.N_ecosystem,
                                        self.N_ecosystem,
                                        self.N_temperatures))## size to define

        for index_action in range(self.N_actions_transition): #for each action
            for (index_temp in range(self.N_temperatures)):
                transition_ecosystem[index_action][0][0][index_temp] = 1 #if coral cover is 0, then stays at 0
                for index_state_start in range(1,self.N_ecosystem):
                    transition_ecosystem[index_action][index_state_start][0][index_temp] = min_prob + norm.cdf(
                                np.log(self.ecosystem_states[0] + 1 / (2 *self.N_ecosystem)),
                                loc=np.log(sik_bar[index_state_start, index_temp, index_action]),
                                scale=self.sigma_eco
                            )
                    for index_state_end in range(1,self.N_ecosystem):
                        transition_ecosystem[index_action][index_state_start][index_state_end][index_temp] = min_prob + norm.cdf(
                                    np.log(self.ecosystem_states[index_state_start] + 1 / (2 *self.N_ecosystem)),
                                    loc=np.log(sik_bar[index_state_start, index_temp, index_action]),
                                    scale=self.sigma_eco
                                )-
                                norm.cdf(
                                        np.log(self.ecosystem_states[index_state_start] - 1 / (2 *self.N_ecosystem)),
                                        loc=np.log(sik_bar[index_state_start, index_temp, index_action]),
                                        scale=self.sigma_eco
                                    )
                    #normalize
                    transition_ecosystem[index_action, index_state_start, :, index_temp] = transition_ecosystem[index_action, index_state_start, :, index_temp]/np.sum(transition_ecosystem[index_action, index_state_start, :, index_temp])

        return transition_ecosystem

    def transition_function_temperatures(self):
        """
        Build transition probability matrix for temperatures.
        transition_temperatures : np.ndarray
            Transition probability matrix of shape (len(time_states), len(temperature_states)).
        """

        transition_temperatures = np.zeros((self.Tmax, self.N_temperatures))

        for t in range(self.Tmax):
            delta_t_avg = temperature_data["Mean"].iloc[t]
            sigma_temp = (temperature_data["Mean"].iloc[t] - temperature_data["X5."].iloc[t]) / norm.ppf(0.95)

            for j in range(self.N_temperatures):
                upper = norm.cdf(self.temperature_states[j] + 1/(2*self.N_temperatures),
                                 loc=delta_t_avg, scale=sigma_temp)
                lower = norm.cdf(self.temperature_states[j] - 1/(2*self.N_temperatures),
                                 loc=delta_t_avg, scale=sigma_temp)

                transition_temperatures[t, j] = upper - lower

            # Normalize row so it sums to 1
            row_sum = transition_temperatures[t, :].sum()
            if row_sum > 0:
                transition_temperatures[t, :] /= row_sum

        return transition_temperatures

    def build_transition_ecosystem_time(self, transition_ecosystem, transition_temperatures):
        """
        Build time-dependent ecosystem transition probabilities.

        Parameters
        ----------
        transition_ecosystem : list of np.ndarray
            Each element is a 3D array of shape (N_ecosystem, N_ecosystem, N_temperatures).
        transition_temperatures : np.ndarray
            Array of shape (N_times, N_temperatures), temperature probabilities at each time.

        Returns
        -------
        transition_ecosystem_time_total : list of np.ndarray
            Each element is a 3D array of shape (N_ecosystem, N_ecosystem, N_times).
        """
        transition_ecosystem_time_total = np.zeros((self.N_actions_transition,
                                                    self.Tmax,
                                                    self.N_ecosystem,
                                                    self.N_ecosystem))

        for action in range(self.N_actions_transition):
            for t in range(self.Tmax):
                temp_probs = transition_temperatures[t, :]  # shape (N_temperatures,)

                for i in range(self.N_ecosystem):
                    for j in range(self.N_ecosystem):
                        transition_ecosystem_time_total[action,t, i, j] = np.sum(
                            temp_probs * transition_ecosystem[action,i, j, :]
                        )

        return transition_ecosystem_time_total

    def transition_function_times(self):
        mat = np.eye(self.Tmax - 1)                       # identity (n-1)x(n-1)
        mat = np.column_stack((np.zeros(self.Tmax - 1), mat))  # prepend col of 0s
        mat = np.vstack((mat, np.append(np.zeros(self.Tmax - 1), 1)))  # add last row [0...0,1]
        return mat
