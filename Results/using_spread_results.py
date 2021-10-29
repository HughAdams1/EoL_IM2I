import pandas as pd
import matplotlib.pyplot as plt
"""
This produces the graph for experiment 2 results
"""
df_1 = pd.read_csv('../CCPPOTrainer_spread_2021-10-29_08-18-39r33_eghz/progress.csv')
df_2 = pd.read_csv('../CCPPOTrainer_spread_2021-10-26_15-44-0477j7unh3/progress.csv')
headers = df_1.columns.values
print(headers)
timesteps = df_1['agent_timesteps_total'][:600]

episode_reward_mean_1 = df_1['policy_reward_mean/ppo_policy_2'][:600]
episode_reward_mean_2 = df_2['policy_reward_mean/ppo_policy_2'][:600]


plt.show()
plt.grid()
plt.xlabel('total episodes')
plt.ylabel('episode reward mean')
plt.plot(timesteps, episode_reward_mean_1, color='red')
plt.plot(timesteps, episode_reward_mean_2, color='blue')

plt.legend(['PPO-IM', 'PPO-cc'])
plt.savefig("spread_results")
plt.show()