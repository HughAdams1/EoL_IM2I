"""I have trained centralised critic PPO on Camber, now I need to extract 1 of those PPO's and run it with an untrained
PPO to see how it fairs
"""

import argparse
import ray
import ppo_mod as ppo
from ray.rllib.examples.models.centralized_critic_models import \
    CentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_reference_v2
from ray.tune.registry import register_env
from bp1_utils import TorchCentralizedCriticModel
#from bp1_utils import CCTrainer as CCTrainer_loaded
from ray.rllib.agents.ppo.ppo import PPOTrainer

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
    help="The DL framework 6.")

if __name__ == "__main__":
    config = ppo.DEFAULT_CONFIG.copy()
    config["env_config"] = {"local_ratio": 0.5, "max_cycles": 25, "continuous_actions": False}
    env = ParallelPettingZooEnv(env_creator(config))
    observation_space = env.observation_space
    action_space = env.action_space
    del env

    args = parser.parse_args()
    ModelCatalog.register_custom_model(
        "cc_model", TorchCentralizedCriticModel
        if args.framework == "torch" else CentralizedCriticModel)

    config["multiagent"] = {
        "env": "spread",
        "policies": {"shared_policy": (None, observation_space, action_space, {
                    "framework": args.framework,
                    }),
                     },
        "policy_mapping_fn": lambda agent_id, episode, **kwargs: "shared_policy",
        "policies_to_train": ["shared_policy"]
    }
    config["log_level"] = "WARN"
    config["num_workers"] = 0
    config["no_done_at_end"] = False
    config["framework"] = args.framework
    config["horizon"] = 100
    config["rollout_fragment_length"] = 10
    config["env"] = "spread"
    config["model"] = {#"custom_model": "cc_model",
                       "fcnet_hiddens": [128, 128],
                       "fcnet_activation": nn.Tanh
                       }
    config["batch_mode"] = "complete_episodes"
    config["use_critic"] = True

    ray.init()

    trainer_loaded = PPOTrainer(config=config, env="spread")
    #trainer_loaded.restore("exp1_stage1_PPO_centralised\checkpoint_001300\checkpoint-1300")
    trainer_loaded.restore("exp1_stage1_PPO_centralised/checkpoint_001700/checkpoint-1700")
    weights = trainer_loaded.get_weights()

    config_student = ppo.DEFAULT_CONFIG.copy()
    config["env_config"] = {"local_ratio": 0.5, "max_cycles": 25, "continuous_actions": False}

    config_student["multiagent"] = {
        "env": "spread",
        "policies": {"shared_policy": (None, observation_space, action_space, {
                    "framework": args.framework,
                    }),
                     "ppo_policy_student": (None, observation_space, action_space, {
                         "framework": args.framework,
                     }),
                     },
        "policy_mapping_fn": lambda agent_id, episode,
                                    **kwargs: "shared_policy" if "0" in agent_id else "ppo_policy_student",
        "policies_to_train": ["ppo_policy_student"]
    }
    config_student["log_level"] = "WARN"
    config_student["num_workers"] = 0
    config_student["no_done_at_end"] = False
    config_student["framework"] = args.framework
    config_student["horizon"] = 100
    config_student["rollout_fragment_length"] = 10
    config_student["env"] = "spread"
    config_student["model"] = {
                       "fcnet_hiddens": [128, 128],
                       "fcnet_activation": nn.ReLU
                       }
    config_student["batch_mode"] = "complete_episodes"
    config_student["use_critic"] = True


    trainer_snt = PPOTrainer(config=config_student, env="spread")
    trainer_snt.set_weights(trainer_loaded.get_weights(["shared_policy"]))

    new_weights = trainer_snt.get_weights()

    results = []

    for i in range(3000):
        result = trainer_snt.train()
        #print("episode", i, "reward_mean:", result["episode_reward_mean"])
        results.append(result)
        if i % 100 == 99:
            checkpoint = trainer_snt.save()
            print("episode", i,"checkpoint", checkpoint)