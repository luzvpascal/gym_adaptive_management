import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AdaptiveManagementBeliefMDP(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a base environment for adaptive management problems with a finite state and action space
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    def __init__(
                self,
                params,
                Tmax=100,
                see_belief=True,
                render_mode="console",
    ):

        super(AdaptiveManagementBeliefMDP, self).__init__()
        self.render_mode = render_mode
        self.see_belief = see_belief

        # Parameters
        self.init_state = params["init_state"]
        self.transition_function = params["transition_function"]
        self.reward_function = params["reward_function"]
        self.Tmax = Tmax

        #
        self.N_actions = self.transition_function.shape[1]
        self.N_states = self.transition_function.shape[2]
        self.N_models = self.transition_function.shape[0]

        #self.N_actions = self.transition_function.unwrapped.shape[1]
        #self.N_states = self.transition_function.unwrapped.shape[2]
        # self.N_models = self.transition_function.unwrapped.shape[0]
        self.vect_of_states = np.arange(self.N_states)
        self.vect_of_models = np.arange(self.N_models)
        self.time_step = 0

        #if the initial belief is not provided, uniform belief is the default.
        if "init_belief" in params:
            self.init_belief = params["init_belief"]
            self.belief = params["init_belief"]
        else:
            self.init_belief = np.ones(self.N_models)/self.N_models
            self.belief = np.ones(self.N_models)/self.N_models

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

        state = int(self.state)  # 0 or 1
        #time_step = int(self.time_step)

        if self.see_belief:
            belief = self.belief.copy().astype(np.float32)
            observation = {"state": state, "belief": belief}
        else:
            observation = {"state": state}

        info = {}

        return observation, info


    def step(self, action):

        # obtain reward
        reward = self.discount_factor**(self.time_step)*self.reward_function[self.state][action]

        #new state
        current_state = self.state
        probabilities = self.belief_MPD_observation_probabilities(action)
        self.state = np.random.choice(self.vect_of_states, 1, p=probabilities)[0]

        #update belief
        self.update_belief(action, self.state,current_state)

        #update time step
        self.time_step += 1


        # Are we at the left of the grid?
        terminated = bool(self.time_step == self.Tmax)
        truncated = False  # we do not limit the number of steps here

        state = int(self.state)
        if self.see_belief:
            belief = self.belief.copy().astype(np.float32)
            observation = {"state": state, "belief": belief}
        else:
            observation = {"state": state}

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def update_belief(self,action,new_observation,past_observation):
        new_belief = np.zeros(self.N_models)
        for k in range(self.N_models):
            new_belief[k] = self.transition_function[k][action][past_observation][new_observation]*self.belief[k]

        new_belief = new_belief/np.sum(new_belief)
        np.clip(new_belief, 0, 1)
        self.belief = new_belief

    def belief_MPD_observation_probabilities(self,action):
        probabilities = np.zeros(self.N_states)
        for k in range(self.N_states):
            for i in range(self.N_models):
                probabilities[k]+=self.transition_function[i][action][self.state][k]*self.belief[i]
        return(probabilities)



    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(self.time_step)
            print(self.state)
            print(self.belief)

    def close(self):
        pass
