"""
This trains PPO with a centralised critic, creating two agents, that we will use as Teachers. It is utilising
my modifications in from ppo_mod and ppo_policy_mod which are taken from rllib's ppo folder.

I have made a small change to rllib.agents.trainer, trainer.allow_unknown_configs = True. This allows me to create the
config "use_intrinsic_imitation".
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
from bp1_utils import CCTrainer
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg import ddpg
from bp1_utils import save_obj


def env_creator(config):
    env = simple_reference_v2.parallel_env(continuous_actions=True)
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

if __name__ == "__main__":
    config = ddpg.DEFAULT_CONFIG.copy()
    config["env_config"] = {"local_ratio": 0.5, "max_cycles": 25, "continuous_actions": True}
    env = ParallelPettingZooEnv(env_creator(config))
    observation_space = env.observation_space
    action_space = env.action_space
    del env

    args = parser.parse_args()

    config["multiagent"] = {
        "env": "spread",
        "policies": {"shared_policy": (None, observation_space, action_space, {
                    "framework": args.framework,
                    }),
                     },
        "policy_mapping_fn": lambda agent_id, episode, **kwargs: "shared_policy",
        "policies_to_train": ["shared_policy"]
    }
    config["actor_hiddens"] = [64, 64]
    config["critic_hiddens"] = [64, 64]
    config["twin_q"] = True
    config["smooth_target_policy"] = True
    #config['simple_optimizer'] = True
    config["log_level"] = "WARN"
    config["critic_lr"] = 0.01
    config["actor_lr"] = 0.01
    config["gamma"] = 0.95
    config["tau"] = 0.01
    config["buffer_size"] = 1000000 #1e6
    config["train_batch_size"] = 1024
    config["num_workers"] = 0
    config["no_done_at_end"] = False
    config["framework"] = args.framework
    config["horizon"] = 100
    config["rollout_fragment_length"] = 100
    #config["training_intensity"] = 100
    config["env"] = "spread"
    config["model"] = {#"custom_model": "cc_model",
                       "fcnet_hiddens": [64, 64],
                       "fcnet_activation": nn.ReLU
                       }
    config["batch_mode"] = "complete_episodes"

    ray.init()

    results = []
    # CCTrainer._allow_unknown_configs = True, I changed this in file and it works
    trainer = DDPGTrainer(config=config, env="spread")
    for i in range(1000):
        result = trainer.train()
        results.append(result)
        print("episode", i, "reward_mean:", result["episode_reward_mean"])
        if i % 100 == 99:
            checkpoint = trainer.save()
            print("checkpoint of episode", i, "saved at", checkpoint)


    #save_obj(results, "s1_baseline_results")

#checkpoint of episode 999 saved at C:\Users\hugha/ray_results\exp1_stage1_DDPG_centralised\checkpoint_001000\checkpoint-1000
