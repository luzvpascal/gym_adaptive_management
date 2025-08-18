import gymnasium as gym
import gym_adaptive_management
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
from gym_adaptive_management.envs.adaptive_management_deployment import TechnoDeployEnv

from scipy.stats import norm

env = TechnoDeployEnv()

print("Scenario:", env.scenario)
print("Data (scenario filtered):\n", env.data_scenario)
print("Temperatures (Mean):", env.data_temperatures.tolist())
print("Number of time steps:", env.N_times)
print("\n--- Model Parameters ---")
print("r:", env.r)
print("Kmin:", env.Kmin)
print("Kmax:", env.Kmax)
print("m:", env.m)
print("sigma_eco:", env.sigma_eco)
print("\n--- MDP Parameters ---")
print("N_ecosystem:", env.N_ecosystem)
print("Ecosystem states:", env.ecosystem_states)
print("Values delta_t_K:", env.values_delta_t_K)
print("N_values_delta_t_K:", env.N_values_delta_t_K)
print("Temperature states:", env.temperature_states)
print("N_temperatures:", env.N_temperatures)
print("N_actions_transition:", env.N_actions_transition)
print("DEP_EFFECT:", env.DEP_EFFECT)

print("=======================================================================")
print("testing each function")
print("=======================================================================")
print("testing K_function")
print("=======================================================================")

delta_t_crit = 1.5
delta_t_values = np.arange(0, 3.1, 0.1)  # from 0 to 3 step 0.1
K_values = []

for dt in delta_t_values:
    K = env.K_function(delta_t_crit, dt)
    K_values.append(K)

# Plot results
plt.plot(delta_t_values, K_values, marker='o', label=f"delta_t_crit = {delta_t_crit}")
plt.xlabel("ΔT (current local temperature)")
plt.ylabel("Carrying capacity K")
plt.title("Carrying Capacity Function")
plt.legend()
plt.grid(True)
plt.show()

print("testing ecosystem_dynamics")
print("=======================================================================")
x0 = 0.1
K = 1
x_time = [x0]
for t in range(50):
    x = env.ecosystem_dynamics(x_time[t], K)
    x_time.append(x)

# Plot results
plt.plot(np.arange(51), x_time, marker='o', label=f"delta_t_crit = {delta_t_crit}")
plt.xlabel("Time")
plt.ylabel("Coral cover")
plt.legend()
plt.grid(True)
plt.show()

print("testing sik_bar_function")
print("=======================================================================")

sik_bar = env.sik_bar_function(delta_t_crit)
print(sik_bar)
print(sik_bar[0][10])
print(sik_bar[2][10])

print("testing transition_function_ecosystem")
print("=======================================================================")

transition_ecosystem = env.transition_function_ecosystem(sik_bar)
index_action = 0
index_temp=0
index_state_start = 2
print(transition_ecosystem[index_action, index_state_start, :, index_temp])
print(transition_ecosystem.shape)

print("testing transition_function_temperatures")
print("=======================================================================")
transition_temperatures = env.transition_function_temperatures()
print(transition_temperatures.shape)

print("testing build_transition_ecosystem_time")
print("=======================================================================")
transition_ecosystem_time = env.build_transition_ecosystem_time(transition_ecosystem, transition_temperatures)
print(transition_ecosystem_time.shape)

print("testing transition_function_times")
print("=======================================================================")
transition_times = env.transition_function_times()
print(transition_times)
print(transition_times.shape)

print("testing transition_function_ecosystem_time")
print("=======================================================================")
transition_function_delta_K = env.transition_function_ecosystem_time(transition_ecosystem_time, transition_times)
print(transition_function_delta_K.shape)
