"""
This trains PPO with a centralised critic, creating two agents, that we will use as Teachers. It is utilising
my modifications in from ppo_mod and ppo_policy_mod which are taken from rllib's ppo folder.

I have made a small change to rllib.agents.trainer, trainer.allow_unknown_configs = True. This allows me to create the
config "use_intrinsic_imitation".
"""

import argparse
import ray
###################################
#import ray.rllib.agents.ppo as ppo
import ppo_mod as ppo
###################################

from ray.rllib.examples.models.centralized_critic_models import \
    CentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_reference_v2
from ray.tune.registry import register_env

from bp1_utils import TorchCentralizedCriticModel as TorchCentralizedCriticModel_pz
from bp1_utils import CCTrainer

def env_creator(config):
    env = simple_reference_v2.parallel_env()
    return env

register_env("spread", lambda config: ParallelPettingZooEnv(env_creator(config)))


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=100,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=7.99,
    help="Reward at which we stop training.")


if __name__ == "__main__":
    config = ppo.DEFAULT_CONFIG.copy()
    config["env_config"] = {"local_ratio": 0.5, "max_cycles": 25, "continuous_actions": False}
    env = ParallelPettingZooEnv(env_creator(config))
    observation_space = env.observation_space
    action_space = env.action_space
    del env

    args = parser.parse_args()
    ModelCatalog.register_custom_model(
        "cc_model", TorchCentralizedCriticModel_pz
        if args.framework == "torch" else CentralizedCriticModel)

    config["multiagent"] = {
        "env": "spread",
        "policies": {"ppo_policy_2": (None, observation_space, action_space, {
                    "framework": args.framework,
                    }),
                     "ppo_policy_1": (None, observation_space, action_space, {
                         "framework": args.framework,
                     })
                     },
        "policy_mapping_fn": lambda agent_id, episode, **kwargs: "ppo_policy_1" if "1" in agent_id else "ppo_policy_2",
        "policies_to_train": ["ppo_policy_1", "ppo_policy_2"]
    }
    config["log_level"] = "WARN"
    config["num_workers"] = 0
    config["no_done_at_end"] = False
    config["framework"] = args.framework
    config["horizon"] = 100
    config["rollout_fragment_length"] = 10
    config["env"] = "spread"
    config["model"] = {"custom_model": "cc_model"}
    config["batch_mode"] = "complete_episodes"
    config["use_intrinsic_imitation"] = False
    ray.init(num_cpus=1)

    results = []
    # CCTrainer._allow_unknown_configs = True, I changed this in file and it works
    trainer = CCTrainer(config=config, env="spread")
    for i in range(100000):
        result = trainer.train()
        results.append(result)
        print("episode_reward_mean:", result["episode_reward_mean"])

        if i % 10000 == 9999:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)