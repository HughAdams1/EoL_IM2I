import pandas as pd
import matplotlib.pyplot as plt

df_1 = pd.read_csv('../exp1_stage2_IM/progress.csv')
df_2 = pd.read_csv('../exp1_stage2_PPO/progress.csv')

timesteps = df_1['episodes_total'][:700]

episode_reward_mean_1 = df_1['policy_reward_mean/ppo_policy_student'][:700]
episode_reward_mean_2 = df_2['policy_reward_mean/ppo_policy_student'][:700]

plt.grid()

plt.plot(timesteps, episode_reward_mean_1, color='red')
plt.plot(timesteps, episode_reward_mean_2, color='blue')
plt.xlabel('total episodes')
plt.ylabel('episode reward mean')
plt.legend(['PPO-IM', 'PPO'])
plt.savefig("ref_training_results")
plt.show()