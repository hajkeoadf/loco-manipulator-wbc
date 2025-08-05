# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import torch
import torch.nn as nn
import torch.optim as optim

from .rsl_actor_critic import ActorCritic
from .rsl_rollout_storage import RolloutStorage


class RSLPPO:
    actor_critic: ActorCritic
    
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 mixing_schedule=[0.5, 2000, 4000], 
                 torque_supervision=True,
                 torque_supervision_schedule=[0.1, 1000, 1000],
                 adaptive_arm_gains=True,
                 min_policy_std=None,
                 dagger_update_freq=20,
                 priv_reg_coef_schedule=[0, 0, 0],
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(
            self.actor_critic.history_encoder.parameters(), 
            lr=learning_rate
        )
        self.priv_reg_coef_schedule = priv_reg_coef_schedule

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.min_policy_std = torch.tensor(min_policy_std, device=self.device) if min_policy_std is not None else None

        self.mixing_schedule = mixing_schedule
        self.torque_supervision = torque_supervision
        self.torque_supervision_schedule = torque_supervision_schedule
        self.adaptive_arm_gains = adaptive_arm_gains
        self.counter = 0

        # adaptive arm gains
        self.default_arm_p_gains = None
        self.default_arm_d_gains = None
        self.default_arm_dof_pos = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, hist_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, hist_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, critic_obs, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self._update_actor_critic_obs(obs, critic_obs)
            actions, actions_log_prob, values, hidden_states = self.actor_critic.act(self.transition.observations, self.transition.observations_history, self.transition.critic_observations, hist_encoding, self.transition.hidden_states)
            self.transition.hidden_states = hidden_states
        else:
            actions, actions_log_prob, values, adaptive_gains = self.actor_critic.act(obs, obs_history, critic_obs, hist_encoding)
        return actions, actions_log_prob, values, adaptive_gains

    def process_env_step(self, rewards, arm_rewards, dones, infos):
        self.transition.rewards = torch.cat([rewards, arm_rewards], dim=-1)
        self.transition.dones = dones
        self.transition.actions = infos['actions']
        self.transition.values = infos['values']
        self.transition.actions_log_prob = infos['actions_log_prob']
        self.transition.action_mean = infos['action_mean']
        self.transition.action_sigma = infos['action_sigma']
        
        # Store arm-specific information for torque supervision
        if self.torque_supervision:
            self.transition.target_arm_torques = infos.get('target_arm_torques', torch.zeros_like(arm_rewards))
            self.transition.current_arm_dof_pos = infos.get('current_arm_dof_pos', torch.zeros_like(arm_rewards))
            self.transition.current_arm_dof_vel = infos.get('current_arm_dof_vel', torch.zeros_like(arm_rewards))

        self.storage.add_transitions(self.transition, self.torque_supervision)
        self.transition.clear()

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_priv_reg_loss = 0
        mean_torque_supervision_loss = 0
        value_mixing_ratio = self.get_value_mixing_ratio()
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, obs_history_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, target_arm_torques_batch, current_arm_dof_pos_batch, current_arm_dof_vel_batch in generator:

            self.actor_critic.act(obs_batch, obs_history_batch, critic_obs_batch, False, hid_states_batch, masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, hid_states_batch, masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptation module update
            priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_history_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedule[2]), 0) / self.priv_reg_coef_schedule[3], 1)
            priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedule[1] - self.priv_reg_coef_schedule[0]) + self.priv_reg_coef_schedule[0]
            # priv_reg_loss = torch.zeros(1, device=self.device)

            # KL
            if self.desired_kl is not None and self.schedule == 'adaptive':
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                            torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                            2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl * 0.5:
                    self.learning_rate = min(1e-1, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # Surrogate loss
            mixing_advantages_batch = torch.zeros_like(advantages_batch)
            mixing_advantages_batch[..., 0] = advantages_batch[..., 0] + value_mixing_ratio * advantages_batch[..., 1]
            mixing_advantages_batch[..., 1] = advantages_batch[..., 1] + value_mixing_ratio * advantages_batch[..., 0]
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(mixing_advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(mixing_advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                              1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                              self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Torque supervision loss
            if self.torque_supervision and target_arm_torques_batch is not None:
                torque_weight = self.get_torque_supervision_weight()
                # Extract arm actions (last 6 dimensions)
                arm_actions = actions_batch[:, -6:]
                # Compute torque supervision loss (simplified)
                torque_supervision_loss = torch.mean((arm_actions - target_arm_torques_batch).pow(2)) * torque_weight
            else:
                torque_supervision_loss = torch.tensor(0.0, device=self.device)

            # Entropy loss
            entropy_loss = -entropy_batch.mean()

            # Total loss
            loss = surrogate_loss \
                + self.value_loss_coef * value_loss \
                + self.entropy_coef * entropy_loss \
                + torque_supervision_loss \
                + priv_reg_coef * priv_reg_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_loss.item()
            mean_torque_supervision_loss += torque_supervision_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_torque_supervision_loss /= num_updates
        mean_priv_reg_loss /= num_updates

        self.storage.clear()
        self.update_counter()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_torque_supervision_loss, mean_priv_reg_loss, priv_reg_coef

    def update_dagger(self):
        """DAgger update for imitation learning"""
        # Implementation for DAgger algorithm
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, obs_history_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, target_arm_torques, current_arm_dof_pos, current_arm_dof_vel, hid_states_batch, masks_batch in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, obs_history_batch, critic_obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_history_batch)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates

        self.storage.clear()
        self.update_counter()

        return mean_hist_latent_loss

    def enforce_min_std(self):
        """Enforce minimum policy standard deviation"""
        if self.min_policy_std is not None:
            self.actor_critic.actor.log_std.data.clamp_(min=torch.log(self.min_policy_std))

    def update_counter(self):
        """Update training counter"""
        self.counter += 1

    def get_value_mixing_ratio(self):
        """Get value mixing ratio for curriculum learning"""
        return min(max((self.counter - self.mixing_schedule[1]) / self.mixing_schedule[2], 0), 1) * self.mixing_schedule[0]

    def get_torque_supervision_weight(self):
        """Get torque supervision weight for curriculum learning"""
        if self.counter < self.torque_supervision_schedule[1]:
            return self.torque_supervision_schedule[0]
        elif self.counter < self.torque_supervision_schedule[2]:
            return self.torque_supervision_schedule[0] + (1.0 - self.torque_supervision_schedule[0]) * (self.counter - self.torque_supervision_schedule[1]) / (self.torque_supervision_schedule[2] - self.torque_supervision_schedule[1])
        else:
            return 1.0

    def set_arm_default_coeffs(self, default_arm_p_gains, default_arm_d_gains, default_arm_dof_pos):
        """Set default arm coefficients for adaptive gains"""
        self.default_arm_p_gains = default_arm_p_gains
        self.default_arm_d_gains = default_arm_d_gains
        self.default_arm_dof_pos = default_arm_dof_pos

    def arm_fk_adaptive_gains(self, delta_arm_p_gains, target_arm_dof_pos, current_arm_dof_pos, current_arm_dof_vel):
        """Adaptive arm gains using forward kinematics"""
        if self.adaptive_arm_gains and self.default_arm_p_gains is not None:
            adaptive_p_gains = self.default_arm_p_gains + delta_arm_p_gains
            return adaptive_p_gains * (target_arm_dof_pos - current_arm_dof_pos) - self.default_arm_d_gains * current_arm_dof_vel
        else:
            return self.arm_fk_fixed_gains(None, target_arm_dof_pos, current_arm_dof_pos, current_arm_dof_vel)

    def arm_fk_fixed_gains(self, _, target_arm_dof_pos, current_arm_dof_pos, current_arm_dof_vel):
        """Fixed arm gains"""
        if self.default_arm_p_gains is not None:
            return self.default_arm_p_gains * (target_arm_dof_pos - current_arm_dof_pos) - self.default_arm_d_gains * current_arm_dof_vel
        else:
            return torch.zeros_like(target_arm_dof_pos) 