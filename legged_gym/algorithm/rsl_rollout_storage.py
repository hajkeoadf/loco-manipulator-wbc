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

import torch
import numpy as np


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.observations_history = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.target_arm_torques = None
            self.current_arm_dof_pos = None
            self.current_arm_dof_vel = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, obs_hist_shape, privileged_obs_shape, actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.observations_history = torch.zeros(num_transitions_per_env, num_envs, *obs_hist_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)  # [leg_reward, arm_reward]
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # Arm-specific storage for torque supervision
        self.target_arm_torques = torch.zeros(num_transitions_per_env, num_envs, 6, device=self.device)
        self.current_arm_dof_pos = torch.zeros(num_transitions_per_env, num_envs, 6, device=self.device)
        self.current_arm_dof_vel = torch.zeros(num_transitions_per_env, num_envs, 6, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition, torque_supervision):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        
        # 修复：将observations_history展平后再复制
        if transition.observations_history is not None:
            # obs_history shape: [num_envs, obs_history_length, num_obs] -> [num_envs, obs_history_length * num_obs]
            obs_history_flat = transition.observations_history.reshape(transition.observations_history.shape[0], -1)
            self.observations_history[self.step].copy_(obs_history_flat)
        
        if self.privileged_observations is not None: 
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob)
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        
        # Store arm-specific information for torque supervision
        if torque_supervision:
            if transition.target_arm_torques is not None:
                self.target_arm_torques[self.step].copy_(transition.target_arm_torques)
            if transition.current_arm_dof_pos is not None:
                self.current_arm_dof_pos[self.step].copy_(transition.current_arm_dof_pos)
            if transition.current_arm_dof_vel is not None:
                self.current_arm_dof_vel[self.step].copy_(transition.current_arm_dof_vel)

        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            self.saved_hidden_states_a = None
            self.saved_hidden_states_c = None
        else:
            self.saved_hidden_states_a = hidden_states[0].clone()
            self.saved_hidden_states_c = hidden_states[1].clone()

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            next_values = next_values * next_non_terminal

            delta = self.rewards[step] + gamma * next_values - self.values[step]
            advantage = delta + gamma * lam * next_non_terminal * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)

        observations = self.observations.flatten(0, 1)
        observations_history = self.observations_history.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        
        # Arm-specific data
        target_arm_torques = self.target_arm_torques.flatten(0, 1)
        current_arm_dof_pos = self.current_arm_dof_pos.flatten(0, 1)
        current_arm_dof_vel = self.current_arm_dof_vel.flatten(0, 1)

        for epoch in range(num_epochs):
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                obs_history_batch = observations_history[batch_idx]
                critic_obs_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                advantages_batch = advantages[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                
                # Arm-specific batches
                target_arm_torques_batch = target_arm_torques[batch_idx]
                current_arm_dof_pos_batch = current_arm_dof_pos[batch_idx]
                current_arm_dof_vel_batch = current_arm_dof_vel[batch_idx]

                yield obs_batch, obs_history_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, None, None, target_arm_torques_batch, current_arm_dof_pos_batch, current_arm_dof_vel_batch

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # Implementation for recurrent networks (if needed)
        pass 