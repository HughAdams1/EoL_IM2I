"""
This is a Novel Exploration module, added in to ray/rllib/utils/exploration
"""

from gym.spaces import Discrete, MultiDiscrete, Space
import numpy as np
from typing import Optional, Tuple, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, MultiCategorical
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, \
    try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_ops import one_hot as tf_one_hot
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
F = None
if nn is not None:
    F = nn.functional

#set 5 for spread, set 50 for reference
action_size = 5

class Imitation(Exploration):
    """Implementation of:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf

    Learns a simplified model of the environment based on three networks:
    1) Embedding observations into latent space ("feature" network).
    2) Predicting the action, given two consecutive embedded observations
        ("inverse" network).
    3) Predicting the next embedded obs, given an obs and action
        ("forward" network).

    The less the agent is able to predict the actually observed next feature
    vector, given obs and action (through the forwards network), the larger the
    "intrinsic reward", which will be added to the extrinsic reward.
    Therefore, if a state transition was unexpected, the agent becomes
    "curious" and will further explore this transition leading to better
    exploration in sparse rewards environments.
    """

    def __init__(self,
                 action_space: Space,
                 *,
                 framework: str,
                 teacher_net_config: Optional[ModelConfigDict] = None,
                 model: ModelV2,
                 beta: float = 0.2,
                 eta: float = 1.0,
                 lr: float = 1e-3,
                 sub_exploration: Optional[FromConfigSpec] = None,
                 **kwargs):
        """Initializes a Curiosity object.

        Uses as defaults the hyperparameters described in [1].

        Args:
             feature_dim (int): The dimensionality of the feature (phi)
                vectors.
             feature_net_config (Optional[ModelConfigDict]): Optional model
                configuration for the feature network, producing feature
                vectors (phi) from observations. This can be used to configure
                fcnet- or conv_net setups to properly process any observation
                space.
             inverse_net_hiddens (Tuple[int]): Tuple of the layer sizes of the
                inverse (action predicting) NN head (on top of the feature
                outputs for phi and phi').
             inverse_net_activation (str): Activation specifier for the inverse
                net.
             forward_net_hiddens (Tuple[int]): Tuple of the layer sizes of the
                forward (phi' predicting) NN head.
             forward_net_activation (str): Activation specifier for the forward
                net.
             beta (float): Weight for the forward loss (over the inverse loss,
                which gets weight=1.0-beta) in the common loss term.
             eta (float): Weight for intrinsic rewards before being added to
                extrinsic ones.
             lr (float): The learning rate for the curiosity-specific
                optimizer, optimizing feature-, inverse-, and forward nets.
             sub_exploration (Optional[FromConfigSpec]): The config dict for
                the underlying Exploration to use (e.g. epsilon-greedy for
                DQN). If None, uses the FromSpecDict provided in the Policy's
                default config.
        """
        if not isinstance(action_space, (Discrete, MultiDiscrete)):
            raise ValueError(
                "Only (Multi)Discrete action spaces supported for Curiosity "
                "so far!")

        super().__init__(
            action_space, model=model, framework=framework, **kwargs)

        if self.policy_config["num_workers"] != 0:
            raise ValueError(
                "Curiosity exploration currently does not support parallelism."
                " `num_workers` must be 0!")

        if teacher_net_config is None:
            teacher_net_config = self.policy_config["model"].copy()
        self.teacher_net_config = teacher_net_config

        self.action_dim = self.action_space.n if isinstance(
            self.action_space, Discrete) else np.sum(self.action_space.nvec)

        self.beta = beta
        self.eta = eta
        self.lr = lr

        if sub_exploration is None:
            raise NotImplementedError
        self.sub_exploration = sub_exploration

        # Creates modules/layers inside the actual ModelV2.
        self._teacher_net = ModelCatalog.get_model_v2(
            self.model.obs_space,
            self.action_space,
            self.action_space.n, # my addition - might be the wrong format
            model_config=self.teacher_net_config,
            framework=self.framework,
            name="teacher_net",
        )

        # This is only used to select the correct action
        self.exploration_submodule = from_config(
            cls=Exploration,
            config=self.sub_exploration,
            action_space=self.action_space,
            framework=self.framework,
            policy_config=self.policy_config,
            model=self.model,
            num_workers=self.num_workers,
            worker_index=self.worker_index,
        )

    @override(Exploration)
    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):
        # Simply delegate to sub-Exploration module.
        return self.exploration_submodule.get_exploration_action(
            action_distribution=action_distribution,
            timestep=timestep,
            explore=explore)

    @override(Exploration)
    def get_exploration_optimizer(self, optimizers):
        # Create, but don't add Adam for curiosity NN updating to the policy.
        # If we added and returned it here, it would be used in the policy's
        # update loop, which we don't want (curiosity updating happens inside
        # `postprocess_trajectory`).
        if self.framework == "torch":
            # my addition
            teacher_params = list(self._teacher_net.parameters())

            # Now that the Policy's own optimizer(s) have been created (from
            # the Model parameters (IMPORTANT: w/o(!) the curiosity params),
            # we can add our curiosity sub-modules to the Policy's Model.
            self.model._teacher_net = \
                self._teacher_net.to(self.device)
            # my addition
            self._optimizer_teach = torch.optim.Adam(
                teacher_params, lr=self.lr
            )
        else:
            print("only running in Torch")

        return optimizers

    @override(Exploration)
    def postprocess_trajectory(self, policy, sample_batch,
                               other_agent_batches = None, agent_id = None, episode = None,
                               tf_sess=None):
        """Calculates predicted teacher action

        Also calculates forward and inverse losses and updates the Imitation
        module on the provided batch using our optimizer.

        ignore TF stuff
        """
        if self.framework != "torch":
            self._postprocess_tf(policy, sample_batch, tf_sess)
        else:
            self._postprocess_torch(policy, sample_batch, other_agent_batches,
                                    agent_id, episode)

    def _postprocess_tf(self, policy, sample_batch, tf_sess):
        # tf1 static-graph: Perform session call on our loss and update ops.
        if self.framework == "tf":
            forward_l2_norm_sqared, _ = tf_sess.run(
                [self._forward_l2_norm_sqared, self._update_op],
                feed_dict={
                    self._obs_ph: sample_batch[SampleBatch.OBS],
                    self._next_obs_ph: sample_batch[SampleBatch.NEXT_OBS],
                    self._action_ph: sample_batch[SampleBatch.ACTIONS],
                })
        # tf-eager: Perform model calls, loss calculations, and optimizer
        # stepping on the fly.
        else:
            forward_l2_norm_sqared, _ = self._postprocess_helper_tf(
                sample_batch[SampleBatch.OBS],
                sample_batch[SampleBatch.NEXT_OBS],
                sample_batch[SampleBatch.ACTIONS],
            )
        # Scale intrinsic reward by eta hyper-parameter.
        sample_batch[SampleBatch.REWARDS] = \
            sample_batch[SampleBatch.REWARDS] + \
            self.eta * forward_l2_norm_sqared

        return sample_batch

    def _postprocess_helper_tf(self, obs, next_obs, actions):
        with (tf.GradientTape()
              if self.framework != "tf" else NullContextManager()) as tape:
            # Push both observations through feature net to get both phis.
            phis, _ = self.model._curiosity_feature_net({
                SampleBatch.OBS: tf.concat([obs, next_obs], axis=0)
            })
            phi, next_phi = tf.split(phis, 2)

            # Predict next phi with forward model.
            predicted_next_phi = self.model._curiosity_forward_fcnet(
                tf.concat(
                    [phi, tf_one_hot(actions, self.action_space)], axis=-1))

            # Forward loss term (predicted phi', given phi and action vs
            # actually observed phi').
            forward_l2_norm_sqared = 0.5 * tf.reduce_sum(
                tf.square(predicted_next_phi - next_phi), axis=-1)
            forward_loss = tf.reduce_mean(forward_l2_norm_sqared)

            # Inverse loss term (prediced action that led from phi to phi' vs
            # actual action taken).
            phi_cat_next_phi = tf.concat([phi, next_phi], axis=-1)
            dist_inputs = self.model._curiosity_inverse_fcnet(phi_cat_next_phi)
            action_dist = Categorical(dist_inputs, self.model) if \
                isinstance(self.action_space, Discrete) else \
                MultiCategorical(
                    dist_inputs, self.model, self.action_space.nvec)
            # Neg log(p); p=probability of observed action given the inverse-NN
            # predicted action distribution.
            inverse_loss = -action_dist.logp(tf.convert_to_tensor(actions))
            inverse_loss = tf.reduce_mean(inverse_loss)

            # Calculate the ICM loss.
            loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss

        # Step the optimizer.
        if self.framework != "tf":
            grads = tape.gradient(loss, self._optimizer_var_list)
            grads_and_vars = [(g, v)
                              for g, v in zip(grads, self._optimizer_var_list)
                              if g is not None]
            update_op = self._optimizer.apply_gradients(grads_and_vars)
        else:
            update_op = self._optimizer.minimize(
                loss, var_list=self._optimizer_var_list)

        # Return the squared l2 norm and the optimizer update op.
        return forward_l2_norm_sqared, update_op

    def _postprocess_torch(self, policy, sample_batch,
                           other_agent_batches = None, agent_id = None, episode = None):
        # Push both observations through feature net to get both phis.
        if other_agent_batches != None:
            pol_to_train = policy.config["multiagent"]["policies_to_train"]
            map_fn = policy.config["multiagent"]["policy_mapping_fn"]
            current_policy = map_fn(agent_id, episode)

            if current_policy in pol_to_train:
                [(_, opponent_batch)] = list(other_agent_batches.values())
                opponent_obs = opponent_batch[SampleBatch.CUR_OBS]
                opponent_actions = opponent_batch[SampleBatch.ACTIONS]
                #import ipdb
                #ipdb.set_trace()
                #if other_agent_batches != None:
                teach_inputs = torch.from_numpy(opponent_batch[SampleBatch.CUR_OBS]
                                                ).float().to(policy.device)
                pred_teach_actions, _ = self.model._teacher_net({SampleBatch.OBS: teach_inputs})

                opp_actions_one_hot = F.one_hot(torch.from_numpy(opponent_batch[SampleBatch.ACTIONS]
                                                ).to(policy.device), num_classes=action_size)

                actions_one_hot = F.one_hot(torch.from_numpy(sample_batch[SampleBatch.ACTIONS]
                                                                 ).to(policy.device), num_classes=action_size)

                forward_l2_norm_sqared = 0.5 * torch.sum(
                    torch.pow(pred_teach_actions - opp_actions_one_hot, 2.0), dim=-1)
                loss = torch.mean(forward_l2_norm_sqared)

                reward_value = torch.sum(
                    torch.pow(pred_teach_actions - actions_one_hot, 2.0), dim=-1)

                # Scale intrinsic reward by eta hyper-parameter.
                sample_batch[SampleBatch.REWARDS] = \
                    sample_batch[SampleBatch.REWARDS] + \
                    self.eta * reward_value.detach().cpu().numpy()

                self._optimizer_teach.zero_grad()
                loss.backward()
                self._optimizer_teach.step()
                #import ipdb
                #ipdb.set_trace()

        # Return the postprocessed sample batch (with the corrected rewards).
        return sample_batch

    def _create_fc_net(self, layer_dims, activation, name=None):
        """Given a list of layer dimensions (incl. input-dim), creates FC-net.

        Args:
            layer_dims (Tuple[int]): Tuple of layer dims, including the input
                dimension.
            activation (str): An activation specifier string (e.g. "relu").

        Examples:
            If layer_dims is [4,8,6] we'll have a two layer net: 4->8 (8 nodes)
            and 8->6 (6 nodes), where the second layer (6 nodes) does not have
            an activation anymore. 4 is the input dimension.
        """
        layers = [
            tf.keras.layers.Input(
                shape=(layer_dims[0], ), name="{}_in".format(name))
        ] if self.framework != "torch" else []

        for i in range(len(layer_dims) - 1):
            act = activation if i < len(layer_dims) - 2 else None
            if self.framework == "torch":
                layers.append(
                    SlimFC(
                        in_size=layer_dims[i],
                        out_size=layer_dims[i + 1],
                        initializer=torch.nn.init.xavier_uniform_,
                        activation_fn=act))
            else:
                layers.append(
                    tf.keras.layers.Dense(
                        units=layer_dims[i + 1],
                        activation=get_activation_fn(act),
                        name="{}_{}".format(name, i)))

        if self.framework == "torch":
            return nn.Sequential(*layers)
        else:
            return tf.keras.Sequential(layers)
