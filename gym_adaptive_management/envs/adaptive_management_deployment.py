import gymnasium as gym
import numpy as np
import pandas as pd
from scipy.stats import norm

from gym_adaptive_management.common.read_IPCC_data import load_IPCC_dataset
from gym_adaptive_management.envs.adaptive_management_base import AdaptiveManagement

class TechnoDeployEnv(AdaptiveManagement):
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

    def __init__(
                self,
                values_delta_t_K=np.array([1,2.5]),
                scenario="SSP2_4_5",
                init_state=0.8,
                C_deploy=0.01,
                init_belief=np.array([0.5,0.5]),
                randomize_initial_belief = False,
                discount_factor=1,
                N_states=11
    ):
            df = load_IPCC_dataset()
            self.scenario = scenario
            self.data_scenario = df[df['scenario'] == self.scenario]
            self.data_temperatures = self.data_scenario['Mean'] #average temperature
            self.N_times = len(self.data_temperatures)

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
            self.C_deploy = C_deploy

            combined_transition_matrix = self.main_transition_function()
            reward_matrix = self.main_reward_function()
            super().__init__(
                params={"init_state": 0,
                        "transition_function": combined_transition_matrix,
                        "reward_function": reward_matrix,
                        "init_belief": init_belief,
                        "randomize_initial_belief": randomize_initial_belief,
                        "discount_factor": discount_factor,
                        "Tmax":self.N_times},
                render_mode="console",
            )

    def K_function(self,delta_t_crit,delta_t):
        """
        calculate the carrying capacity for a given delta_t_crit at temperature delta_t
        delta_t_crit: shifting temperature
        delta_t: current local temperature
        """
        return self.Kmin + (self.Kmax - self.Kmin)*(1-1/(1+np.exp(-self.m*(delta_t-delta_t_crit))))

    def ecosystem_dynamics(self,x_t,K):
        """
        calculate the carrying capacity for a given delta_t_crit at temperature delta_t
        delta_t_crit: shifting temperature
        delta_t: current local temperature
        """
        return x_t + (x_t*self.r*(1-x_t/K))

    def sik_bar_function(self, delta_t_crit):
        """
        calculate the average coral cover at the next time step
        for each action, current coral cover, temperature, and ecosystem response (delta_t_crit)

        output:
        sik_bar : np.ndarray
            A 3D array of shape
            (N_actions_transition, N_ecosystem, N_temperatures) containing
            average ecosystem coral cover for each combination of
            ecosystem state, temperature state, and management action.
        """
        sik_bar = np.zeros((self.N_actions_transition, self.N_ecosystem, self.N_temperatures))
        for index_action in range(self.N_actions_transition):
            for index_state in range(self.N_ecosystem):
                for index_temp in range(self.N_temperatures):
                    K_eff = self.K_function(delta_t_crit, self.temperature_states[index_temp]-self.DEP_EFFECT[index_action])
                    sik_bar[index_action,
                            index_state,
                            index_temp] = max(0, self.ecosystem_dynamics(self.ecosystem_states[index_state],
                                                                K_eff))
        return sik_bar

    def transition_function_ecosystem(self, sik_bar):
        """
        Construct the ecosystem state transition probability tensor
        conditional on management actions, temperature states, and
        ecosystem states.

        This function builds a 4D transition probability array:
        (N_actions_transition × N_ecosystem × N_ecosystem × N_temperatures).

        Each slice corresponds to the probability of transitioning from one
        ecosystem state to another, given:
          - a particular management action,
          - a current ecosystem state,
          - and a temperature state.

        Probabilities are computed using the cumulative distribution
        function (CDF) of a log-normal distribution, parameterized by
        `sik_bar` and `sigma_eco`. A small baseline probability
        (`min_prob`) is added to each transition to avoid zero-probability
        events. Rows are normalized so that transition probabilities
        sum to 1.

        Parameters
        ----------
        sik_bar : np.ndarray
            A 3D array of shape
            (N_actions_transition, N_ecosystem, N_temperatures) containing
            average ecosystem coral cover for each combination of
            ecosystem state, temperature state, and management action.

        Returns
        -------
        transition_ecosystem : np.ndarray
            A 4D array of shape
            (N_actions_transition, N_ecosystem, N_ecosystem, N_temperatures),
            where each element represents the transition probability from
            a starting ecosystem state to an ending ecosystem state under
            a given action and temperature.

        Notes
        -----
        - State 0 (zero coral cover) is absorbing: once the ecosystem reaches
          state 0, it remains there with probability 1.
        """
        min_prob = 0.001
        transition_ecosystem = np.zeros((self.N_actions_transition,
                                        self.N_ecosystem,
                                        self.N_ecosystem,
                                        self.N_temperatures))## size to define

        for index_action in range(self.N_actions_transition): #for each action
            for index_temp in range(self.N_temperatures):
                transition_ecosystem[index_action][0][0][index_temp] = 1 #if coral cover is 0, then stays at 0
                for index_state_start in range(1,self.N_ecosystem):
                    if sik_bar[index_action, index_state_start, index_temp]>0: #check that sik_bar is positive
                        transition_ecosystem[index_action][index_state_start][0][index_temp] = min_prob + norm.cdf(
                                    np.log(self.ecosystem_states[0] + 1 / (2 *self.N_ecosystem)),
                                    loc=np.log(sik_bar[index_action, index_state_start, index_temp]),
                                    scale=self.sigma_eco
                                )
                    else:
                        transition_ecosystem[index_action][index_state_start][0][index_temp] = 1#if sik_bar, transition to coral cover 0

                    for index_state_end in range(1,self.N_ecosystem):
                        if sik_bar[index_action, index_state_start, index_temp]>0: #check that sik_bar is positive
                            transition_ecosystem[index_action][index_state_start][index_state_end][index_temp] = min_prob + norm.cdf(
                                        np.log(self.ecosystem_states[index_state_end] + 1 / (2 *self.N_ecosystem)),
                                        loc=np.log(sik_bar[index_action, index_state_start, index_temp]),
                                        scale=self.sigma_eco
                                    )-norm.cdf(
                                            np.log(self.ecosystem_states[index_state_end] - 1 / (2 *self.N_ecosystem)),
                                            loc=np.log(sik_bar[index_action, index_state_start, index_temp]),
                                            scale=self.sigma_eco
                                        )
                        else:#add a small probability
                            transition_ecosystem[index_action][index_state_start][index_state_end][index_temp] = min_prob
                        transition_ecosystem[index_action][index_state_start][index_state_end][index_temp] = round(transition_ecosystem[index_action][index_state_start][index_state_end][index_temp], 4)
                    #normalize
                    transition_ecosystem[index_action, index_state_start, :, index_temp] = transition_ecosystem[index_action, index_state_start, :, index_temp]/np.sum(transition_ecosystem[index_action, index_state_start, :, index_temp])

        return transition_ecosystem

    def transition_function_temperatures(self):
        """
        Build transition probability matrix for temperatures.
        transition_temperatures : np.ndarray
            Transition probability matrix of shape (N_times, N_temperatures).
        """

        transition_temperatures = np.zeros((self.N_times, self.N_temperatures))

        for t in range(self.N_times):
            delta_t_avg = self.data_scenario["Mean"].iloc[t]
            sigma_temp = (self.data_scenario["Mean"].iloc[t] - self.data_scenario["X5."].iloc[t]) / norm.ppf(0.95)

            for j in range(self.N_temperatures):
                upper = norm.cdf(self.temperature_states[j] + 1/(2*self.N_temperatures),
                                 loc=delta_t_avg, scale=sigma_temp)
                lower = norm.cdf(self.temperature_states[j] - 1/(2*self.N_temperatures),
                                 loc=delta_t_avg, scale=sigma_temp)

                transition_temperatures[t, j] = round(upper - lower, 4)

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
            Each element is a 4D array of shape (N_actions_transition, N_ecosystem, N_ecosystem, N_temperatures).
        transition_temperatures : np.ndarray
            Array of shape (N_times, N_temperatures), temperature probabilities at each time.

        Returns
        -------
        transition_ecosystem_time_total : list of np.ndarray
            Each element is a 4D array of shape (N_actions_transition, N_ecosystem, N_ecosystem, N_times).
        """
        transition_ecosystem_time_total = np.zeros((self.N_actions_transition,
                                                    self.N_ecosystem,
                                                    self.N_ecosystem,
                                                    self.N_times))

        for action in range(self.N_actions_transition):
            for t in range(self.N_times):
                temp_probs = transition_temperatures[t, :]  # shape (N_temperatures,)

                for i in range(self.N_ecosystem):
                    for j in range(self.N_ecosystem):
                        transition_ecosystem_time_total[action, i, j,t] = np.sum(
                            temp_probs * transition_ecosystem[action,i, j, :]
                        )

        return transition_ecosystem_time_total

    def transition_function_times(self):
        """
        transition function between time steps
        [0,1,0,...,  0 ,
         0,0,1,0, ..,0,
         ...,
         ..., 0, 0, 1, 0,
         ..., 0, 0, 0, 1,
         ..., 0, 0, 0, 1]
        """
        mat = np.eye(self.N_times - 1)                       # identity (n-1)x(n-1)
        mat = np.column_stack((np.zeros(self.N_times - 1), mat))  # prepend col of 0s
        mat = np.vstack((mat, np.append(np.zeros(self.N_times - 1), 1)))  # add last row [0...0,1]
        return mat

    def tuple_to_index(self, time_step, eco_state_index):
        return((time_step) * self.N_ecosystem + eco_state_index)

    def index_to_tuple(self, index):
        t = index // self.N_ecosystem   # integer division
        i = index % self.N_ecosystem
        return (t, i)

    def index_to_year(self, index):
        """
        Extract the time index from flat index.
        """
        t, _ = self.index_to_tuple(index)
        return t

    def index_to_eco(index):
        """
        Extract the ecosystem index from flat index.
        """
        _, i = self.index_to_tuple(index)
        return i

    def transition_function_ecosystem_time(self, transition_ecosystem_time, transition_times):
        """
        Combine ecosystem transitions with time transitions.

        Parameters
        ----------
        transition_ecosystem_time : list of np.ndarray
            Each element is a 4D array of shape (N_actions_transition, N_ecosystem, N_ecosystem, N_times).
        transition_times : np.ndarray
            Shape (N_times, N_times).

        Returns
        -------
        combined_transition_matrix : np.ndarray
            Shape (N_actions_transition, N_ecosystem*N_times, N_ecosystem*N_times).
        """
        combined_transition_matrix = np.zeros(
            (self.N_actions_transition, self.N_ecosystem * self.N_times, self.N_ecosystem * self.N_times)
        )

        for a in range(self.N_actions_transition):
            current_transition_ecosystem_time = transition_ecosystem_time[a]

            for t in range(self.N_times):
                time_prob = transition_times[t, :]

                # Only loop over nonzero time transitions
                for p in np.where(time_prob > 0)[0]:
                    for i in range(self.N_ecosystem):
                        start_index = self.tuple_to_index(t, i)

                        for j in range(self.N_ecosystem):
                            end_index = self.tuple_to_index(p, j)
                            ecosystem_prob = current_transition_ecosystem_time[i, j, t]

                            combined_transition_matrix[a, start_index, end_index] = round(
                                ecosystem_prob * time_prob[p], 4)

                        # Normalize row so it sums to 1
                        row_sum = combined_transition_matrix[a, start_index, :].sum()
                        if row_sum > 0:
                            combined_transition_matrix[a, start_index, :] /= row_sum

        return combined_transition_matrix

    def main_transition_function(self):
        """
        Compute the transition function for the current case study
        Output:
        combined_transition_matrix : np.ndarray
            Shape (N_values_delta_t_K, N_actions_transition, N_ecosystem*N_times, N_ecosystem*N_times).
        """
        # Allocate result array
        combined_transition_matrix = np.zeros((
                self.N_values_delta_t_K,
                self.N_actions_transition,
                self.N_ecosystem * self.N_times,
                self.N_ecosystem * self.N_times
        ))

        for index_delta_K in range(self.N_values_delta_t_K):
            #compute average coral cover
            sik_bar = self.sik_bar_function(self.values_delta_t_K[index_delta_K])

            ## transitions function of ecosystem states with temperature ####
            transition_ecosystem = self.transition_function_ecosystem(sik_bar)

            ## transition function of temperature with time ####
            transition_temperatures = self.transition_function_temperatures()

            #combine transition ecosystem with time state ####
            transition_ecosystem_time = self.build_transition_ecosystem_time(transition_ecosystem,
                                                               transition_temperatures)
            ## transition function of time ####
            transition_times = self.transition_function_times()

            ## transition function of tupple ecosystem state x time state ####
            combined_matrix = self.transition_function_ecosystem_time(transition_ecosystem_time,
                                                                   transition_times)

            # Store result
            combined_transition_matrix[index_delta_K] = combined_matrix

        return combined_transition_matrix

    def main_reward_function(self):
        """
        reward is b.s-c.a**2
        b: benefit
        s: ecosystem state
        c: deployment costs
        a: intensity of mitigation level

        Outputs reward function as a matrix
        reward_matrix : np.ndarray
            Array of shape (N_ecosystem * N_times, N_actions) with rewards for each
            (ecosystem × time) state–action pair.
        """
        # Initialize reward matrix
        reward_matrix = np.zeros((self.N_ecosystem * self.N_times, self.N_actions_transition))

        # Base reward (BAU = Business As Usual)
        reward_BAU = np.array(self.ecosystem_states)

        # Loop over actions
        for action_index in range(self.N_actions_transition):
            reward_action = reward_BAU - (self.DEP_EFFECT[action_index] ** 2) * self.C_deploy

            # Fill reward matrix for all times (except last absorbing time state)
            for t in range(self.N_times - 1):
                for i in range(self.N_ecosystem):
                    index = self.tuple_to_index(t, i)
                    reward_matrix[index, action_index] = reward_action[i]

        return reward_matrix
