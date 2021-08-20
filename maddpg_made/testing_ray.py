
import ray
import ray.rllib.agents.ppo as ppo
import os
import torch
#initialise ray
#ray.shutdown()
#ray.init(ignore_reinit_error=True)
from ray.tune.logger import pretty_print
ray.init()

#this gets up the dashboard
#print("Dashboard URL: http://{}".format(ray.get_webui_url()))
"""
#set up a directory for saving
import shutil

CHECKPOINT_ROOT = "tmp/ppo/taxi"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
os.environ['RR'] = "C:/Users/hugha/PycharmProjects/pythonProject"
ray_results = os.getenv("RR") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
"""
#instantiate environment
SELECT_ENV = "Taxi-v3"

config = ppo.DEFAULT_CONFIG.copy()
#config["log_level"] = "WARN"
config["framework"] = "torch"
#instatiate agent
agent = ppo.PPOTrainer(config, env=SELECT_ENV)
#ppo.ppo_torch_policy

for i in range(100):
   # Perform one iteration of training the policy with PPO
   result = agent.train()
   print(pretty_print(result))