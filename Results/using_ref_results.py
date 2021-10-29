import pandas as pd
import matplotlib.pyplot as plt

df_1 = pd.read_csv('../exp1_stage1_PPO_cc/progress.csv')
df_2 = pd.read_csv('../exp1_stage1_PPO/progress.csv')
df_3 = pd.read_csv('../exp1_stage1_PPO_IM/progress.csv')
df_4 = pd.read_csv('../exp1_stage1_DDPG_centralised/progress.csv')
df_5 = pd.read_csv('../exp1_stage1_PPO_centralised/progress.csv')


timesteps = df_1['episodes_total'][:1000]

episode_reward_mean_1 = df_1['policy_reward_mean/ppo_policy_2'][:1000]
episode_reward_mean_2 = df_2['policy_reward_mean/ppo_policy_2'][:1000]
episode_reward_mean_3 = df_3['policy_reward_mean/ppo_policy_2'][:1000]
episode_reward_mean_4 = df_4['policy_reward_mean/shared_policy'][:1000]
episode_reward_mean_5 = df_5['policy_reward_mean/shared_policy'][:1000]



plt.grid()

plt.plot(timesteps, episode_reward_mean_1, color='red')
plt.plot(timesteps, episode_reward_mean_2, color='blue')
plt.plot(timesteps, episode_reward_mean_3, color='green')
plt.plot(timesteps, episode_reward_mean_4, color='orange')
plt.plot(timesteps, episode_reward_mean_5, color='brown')
plt.xlabel('total episodes')
plt.ylabel('episode reward mean')
plt.legend(['PPO-cc', 'PPO', 'PPO-IM', 'DDPG', 'PPO-shared-policy'])
plt.savefig("ref_training_results")
plt.show()