"""An example of customizing PPO to leverage a centralized critic.
Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.

"""

import argparse
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
import torch.nn.functional as F
###############################################
#from ray.rllib.agents.ppo.ppo import PPOTrainer
from ppo_mod import PPOTrainer
###############################################
from ray.rllib.models.catalog import ModelCatalog

#######################################################################
from ppo_policy_mod import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
#######################################################################

from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_reference_v2
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.modelv2 import ModelV2

def env_creator(config):
    env = simple_reference_v2.parallel_env()
    return env

register_env("spread", lambda config: ParallelPettingZooEnv(env_creator(config)))


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF. Changed to fit petting zoo
    obs and action spaces. Also includes a model of teacher"""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # Base of the model
        self.model = TorchFC(obs_space, action_space, num_outputs,
                             model_config, name)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = 92  # len(obs_space) + len(obs_space) + len(action_space)  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 32, activation_fn=nn.Tanh),
            SlimFC(32, 1),
        )
        self.model_of_teacher = nn.Sequential(
            SlimFC(21, 64, activation_fn=nn.Tanh),
            SlimFC(64, 50),
        )
        #self._teacher_net = TorchFC(obs_space, action_space, action_space.n,
        #                     model_config, name)
        #teacher_params = list(self.model_of_teacher.parameters())
        #self.teacher_optimizer = torch.optim.Adam(teacher_params, lr=0.01) # add in learning rate

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        input_ = torch.cat([
            obs, opponent_obs,
            torch.nn.functional.one_hot(opponent_actions.long(), 50).float()  # changed from 2 which was for Twostepgame
        ], 1)
        return torch.reshape(self.central_vf(input_), [-1])

    def predict_teacher_action_v2(self, opponent_obs):
        mot_out, _ = self._teacher_net(opponent_obs)
        return mot_out

    def predict_teacher_action(self, opponent_obs):
        #for param in self.model_of_teacher.parameters():
        #    param.requires_grad = True
        mot_out = self.model_of_teacher(opponent_obs)
        return mot_out

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used

    def update_teacher_net(self, loss):
        self.teacher_optimizer.zero_grad()
        loss.backward()
        self.teacher_optimizer.step()


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function)
        else:
            self.compute_central_vf = self.model.central_value_function

class TeacherPredictMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_teacher_prediction = make_tf_callable(self.get_session())(
                self.model.predict_teacher_action_v2)
        else:
            self.compute_teacher_prediction = self.model.predict_teacher_action_v2

class TeacherTrainMixin:

    def __init__(self):
        if self.config["framework"] != "torch":
            self.update_teacher_pol = make_tf_callable(self.get_session())(
                self.model.update_teacher_net)
        else:
            self.update_teacher_pol = self.model.update_teacher_net


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        [(_, opponent_batch)] = list(other_agent_batches.values())
        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(
                sample_batch[SampleBatch.CUR_OBS], policy.device),
            convert_to_torch_tensor(
                sample_batch[OPPONENT_OBS], policy.device),
            convert_to_torch_tensor(
                sample_batch[OPPONENT_ACTION], policy.device)) \
            .cpu().detach().numpy()
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch

def imitation_learning_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    print(sample_batch)
    import ipdb
    ipdb.set_trace()
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_teacher_prediction")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        [(_, opponent_batch)] = list(other_agent_batches.values())
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS])

    TeacherPredictMixin.__init__(policy)
    TeacherTrainMixin.__init__(policy)

    #teach_inputs = convert_to_torch_tensor(
    #    sample_batch[OPPONENT_OBS], policy.device).requires_grad_(True)
    #pred_teacher_actions = policy.compute_teacher_prediction(teach_inputs)

    teach_inputs = torch.from_numpy(
        sample_batch[OPPONENT_OBS]).float().to(policy.device).requires_grad_(True)

    pred_teacher_actions = policy.compute_teacher_prediction({SampleBatch.OBS: teach_inputs})

    if any(sample_batch[OPPONENT_ACTION]) != 0:
        opp_actions_one_hot = F.one_hot(
            convert_to_torch_tensor(sample_batch[OPPONENT_ACTION], policy.device),
            num_classes=50).float().requires_grad_(True)
        #print("teach_inputs:", teach_inputs.requires_grad)
        #print("pred_teacher_actions:", pred_teacher_actions)
        forward_l2_norm_sqared = 0.5 * torch.sum(
             torch.pow(pred_teacher_actions - opp_actions_one_hot, 2.0), dim=-1)
        forward_loss = torch.mean(forward_l2_norm_sqared)
        #print("teach_inputs:", teach_inputs.requires_grad)
        #print("pred_teacher_actions:", pred_teacher_actions.requires_grad)
        #print("opp_actions_one_hot:", opp_actions_one_hot.requires_grad)
        #print("forward_loss:", forward_loss.requires_grad)

        sample_batch[SampleBatch.REWARDS] = \
            sample_batch[SampleBatch.REWARDS] + \
            forward_l2_norm_sqared.detach().cpu().numpy()

        policy.update_teacher_pol(forward_loss)
        # Calculate the ICM loss.
        #loss = forward_loss
        # Perform an optimizer step.
        #self._optimizer.zero_grad()
        #loss.backward()
        #self._optimizer.step()

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch







# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss if not policy.config["framework"] == "torch" else torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION])
    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    #TeacherPredictMixin.__init__(policy)



def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])
    #TeacherPredictMixin.__init__(policy)



def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out),
    }

CCPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CCPPOTFPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_tf_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin, TeacherPredictMixin
    ])

CCPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="CCPPOTorchPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_init=setup_torch_mixins,
    mixins=[
        TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
        CentralizedValueMixin, TeacherPredictMixin
    ])

def get_policy_class(config):
    if config["framework"] == "torch":
        return CCPPOTorchPolicy

CCTrainer = PPOTrainer.with_updates(
    name="CCPPOTrainer",
    default_policy=CCPPOTorchPolicy,
    get_policy_class=get_policy_class,
)

