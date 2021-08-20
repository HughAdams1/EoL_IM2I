# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import ray
import ray.rllib
import pettingzoo
from pettingzoo.mpe import simple_spread_v2
from ray import tune
from ray.rllib.agents.


#local mode good for debugging - change when running experiments
ray.init(local_mode=True)

tune.run(PPOTrainer, config={"env": "CartPole-v0"})

env = simple_spread_v2.parallel_env()
def simple_policy(agent):
    action = env.action_spaces[agent].sample()
    return action

observations = env.reset()

max_cycles = 500
for step in range(max_cycles):
    env.render(mode='human')
    actions = {agent: simple_policy(agent) for agent in env.agents}
    observations, rewards, dones, infos = env.step(actions)
env.close()



