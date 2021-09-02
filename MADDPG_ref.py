import os
import ray
import ray.rllib.agents.maddpg as maddpg
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_reference_v2
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.examples.policy.random_policy import RandomPolicy

def env_creator(config):
    env = simple_reference_v2.parallel_env()
    return env


config = a3c.DEFAULT_CONFIG.copy()
config["env_config"] = {"local_ratio": 0.5, "max_cycles": 25, "continuous_actions": False}

register_env("spread", lambda config: ParallelPettingZooEnv(env_creator(config)))


env = ParallelPettingZooEnv(env_creator(config))
observation_space = env.observation_space
action_space = env.action_space
del env

config["multiagent"] = {
    "env": "spread",
    "policies": {"a3c_policy": (None, observation_space, action_space, {}), # None at the front then uses default policy
                 "random": (RandomPolicy, observation_space, action_space, {}),
                 "ppo_policy": (None, observation_space, action_space, {})},
    "policy_mapping_fn": lambda agent_id, episode, **kwargs: "a3c_policy",
    "policies_to_train": ["a3c_policy"]
}

#config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
config["log_level"] = "WARN"
config["num_workers"] = 1
config["no_done_at_end"] = False
config["framework"] = "torch"
config["horizon"] = 100
config["rollout_fragment_length"] = 10
config["env"] = "spread"
# alg_name = "A3C"
ray.init(num_cpus=1)
trainer = a3c.A3CTrainer(config=config, env="spread")

results = []

for i in range(100):
    result = trainer.train()
    results.append(result)
    print(pretty_print(result))