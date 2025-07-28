import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AdaptiveManagement(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a base environment for adaptive management problems
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    DO_NOTHING = 0
    INVEST = 1

    IDLE = 0
    READY = 1

    def __init__(self, transition_function, reward_function, #true_model_index=0,
                 Tmax=100, render_mode="console"):
        super(AdaptiveManagement, self).__init__()
        self.render_mode = render_mode

        # Size of the 1D-grid
        N_states = 2
        # Initialize the agent at the right of the grid
        self.state = self.IDLE

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        N_actions = 2
        self.action_space = spaces.Discrete(N_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.transition_function = transition_function
        self.N_transition_function = self.transition_function.unwrapped.shape[0]
        self.reward_function = reward_function

        self.belief = np.ones(self.N_transition_function)/self.N_transition_function
        self.time_step = 0

        self.true_model_index = np.random.choice([0,1], 1, p=self.belief)[0]
        self.true_transition_model = self.transition_function[self.true_model_index]

        self.Tmax = Tmax

        self.observation_space = spaces.Dict({
                                        "state": spaces.Discrete(2),
                                        "belief": spaces.Box(0, 1, shape=(self.N_transition_function,))
                                        #,
                                        #"time_step": spaces.Discrete(self.Tmax)
                                        })

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed, options=options)

        self.state = self.IDLE
        self.belief = np.ones(self.N_transition_function) / self.N_transition_function
        self.time_step = 0

        state = int(self.state)  # 0 or 1
        belief = self.belief.copy().astype(np.float32)
        #time_step = int(self.time_step)

        observation = {"state": state, "belief": belief
                       #, "time_step":time_step
                       }
        info = {}

        if self.true_model_index ==1:
            self.true_model_index = 0
        else:
            self.true_model_index = 1

        #self.true_model_index = np.random.choice([0,1], 1, p=self.belief)[0]

        self.true_transition_model = self.transition_function[self.true_model_index]

        return observation, info


    def step(self, action):
        if action == self.DO_NOTHING or action==self.INVEST:
            # obtain reward
            reward = self.reward_function[self.state][action]

            #new state
            elements = [self.IDLE, self.READY]
            current_state = self.state
            probabilities = self.true_transition_model[action][current_state]
            self.state = np.random.choice(elements, 1, p=probabilities)[0]

            #update belief
            self.update_belief(action, self.state,current_state)

            #update time step
            self.time_step += 1
        else:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )

        # Are we at the left of the grid?
        terminated = bool(self.time_step == self.Tmax)
        truncated = False  # we do not limit the number of steps here

        state = int(self.state)  # 0 or 1
        belief = self.belief.copy().astype(np.float32)
        time_step = int(self.time_step)
        observation = {"state": state, "belief": belief
                       #, "time_step":time_step
                       }

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

      new_belief = np.zeros(self.N_transition_function)
      for k in range(self.N_transition_function):
        new_belief[k] = self.transition_function[k][action][past_observation][new_observation]*self.belief[k]

      new_belief = new_belief/np.sum(new_belief)
      np.clip(new_belief, 0, 1)
      self.belief = new_belief


    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(self.time_step)
            print(self.state)
            print(self.belief)

    def close(self):
        pass
