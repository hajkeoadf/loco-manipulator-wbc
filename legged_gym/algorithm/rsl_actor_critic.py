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

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


# History Encoder
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class ActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        priv_encoder_dims=[64, 20],
                        activation='elu',
                        init_std=1,
                        num_hist=10,
                        num_prop=48,
                        **kwargs):
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        # History encoder
        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 64)

        # Privileged information encoder
        self.priv_encoder = nn.Sequential(
            nn.Linear(num_actor_obs - num_prop, priv_encoder_dims[0]), activation,
            nn.Linear(priv_encoder_dims[0], priv_encoder_dims[1]), activation
        )

        # Actor with separate control heads
        class Actor(nn.Module):
            def __init__(self, mlp_input_dim_a, actor_hidden_dims, activation, 
                         leg_control_head_hidden_dims, arm_control_head_hidden_dims,
                         num_leg_actions, num_arm_actions, adaptive_arm_gains, 
                         adaptive_arm_gains_scale, num_priv, num_hist, num_prop, priv_encoder_dims):
                super(Actor, self).__init__()
                
                self.adaptive_arm_gains = adaptive_arm_gains
                self.adaptive_arm_gains_scale = adaptive_arm_gains_scale
                
                # Shared layers
                shared_layers = []
                shared_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
                shared_layers.append(activation)
                for l in range(len(actor_hidden_dims)):
                    if l == len(actor_hidden_dims) - 1:
                        break
                    shared_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    shared_layers.append(activation)
                self.shared = nn.Sequential(*shared_layers)
                
                # Leg control head
                leg_layers = []
                leg_layers.append(nn.Linear(actor_hidden_dims[-1], leg_control_head_hidden_dims[0]))
                leg_layers.append(activation)
                for l in range(len(leg_control_head_hidden_dims)):
                    if l == len(leg_control_head_hidden_dims) - 1:
                        leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], num_leg_actions))
                        break
                    leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
                    leg_layers.append(activation)
                self.leg_head = nn.Sequential(*leg_layers)
                
                # Arm control head
                arm_layers = []
                arm_layers.append(nn.Linear(actor_hidden_dims[-1], arm_control_head_hidden_dims[0]))
                arm_layers.append(activation)
                for l in range(len(arm_control_head_hidden_dims)):
                    if l == len(arm_control_head_hidden_dims) - 1:
                        arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], num_arm_actions))
                        break
                    arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
                    arm_layers.append(activation)
                self.arm_head = nn.Sequential(*arm_layers)
                
                # Adaptive arm gains head
                if adaptive_arm_gains:
                    self.adaptive_gains_head = nn.Sequential(
                        nn.Linear(actor_hidden_dims[-1], 64), activation,
                        nn.Linear(64, 6)  # 6 arm joints
                    )
                
                # Action std
                self.log_std = nn.Parameter(torch.ones(num_actions) * np.log(init_std))

            def forward(self, obs, hist_encoding=False):
                shared_features = self.shared(obs)
                
                # Leg actions
                leg_actions = self.leg_head(shared_features)
                
                # Arm actions
                arm_actions = self.arm_head(shared_features)
                
                # Combine actions
                actions = torch.cat([leg_actions, arm_actions], dim=-1)
                
                # Adaptive gains (if enabled)
                if self.adaptive_arm_gains:
                    adaptive_gains = self.adaptive_gains_head(shared_features) * self.adaptive_arm_gains_scale
                else:
                    adaptive_gains = None
                
                return actions, adaptive_gains

            def infer_priv_latent(self, obs):
                return self.priv_encoder(obs)

            def infer_hist_latent(self, obs):
                return self.history_encoder(obs)

        # Critic with separate value heads
        class Critic(nn.Module):
            def __init__(self, mlp_input_dim_c, critic_hidden_dims, activation,
                         leg_control_head_hidden_dims, arm_control_head_hidden_dims,
                         num_priv, num_hist, num_prop):
                super(Critic, self).__init__()
                
                # Shared layers
                shared_layers = []
                shared_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
                shared_layers.append(activation)
                for l in range(len(critic_hidden_dims)):
                    if l == len(critic_hidden_dims) - 1:
                        break
                    shared_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    shared_layers.append(activation)
                self.shared = nn.Sequential(*shared_layers)
                
                # Leg value head
                leg_layers = []
                leg_layers.append(nn.Linear(critic_hidden_dims[-1], leg_control_head_hidden_dims[0]))
                leg_layers.append(activation)
                for l in range(len(leg_control_head_hidden_dims)):
                    if l == len(leg_control_head_hidden_dims) - 1:
                        leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], 1))
                        break
                    leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
                    leg_layers.append(activation)
                self.leg_value_head = nn.Sequential(*leg_layers)
                
                # Arm value head
                arm_layers = []
                arm_layers.append(nn.Linear(critic_hidden_dims[-1], arm_control_head_hidden_dims[0]))
                arm_layers.append(activation)
                for l in range(len(arm_control_head_hidden_dims)):
                    if l == len(arm_control_head_hidden_dims) - 1:
                        arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], 1))
                        break
                    arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
                    arm_layers.append(activation)
                self.arm_value_head = nn.Sequential(*arm_layers)

            def forward(self, obs):
                shared_features = self.shared(obs)
                
                # Separate value estimates
                leg_value = self.leg_value_head(shared_features)
                arm_value = self.arm_value_head(shared_features)
                
                # Combine values
                values = torch.cat([leg_value, arm_value], dim=-1)
                
                return values

        # Initialize actor and critic
        mlp_input_dim_a = num_actor_obs + 64 + priv_encoder_dims[1]  # obs + hist + priv
        mlp_input_dim_c = num_critic_obs + 64 + priv_encoder_dims[1]  # obs + hist + priv
        
        self.actor = Actor(mlp_input_dim_a, actor_hidden_dims, activation,
                          [128, 64], [128, 64], 8, 6,  # leg and arm action dimensions
                          True, 0.1, priv_encoder_dims[1], num_hist, num_prop, priv_encoder_dims)
        
        self.critic = Critic(mlp_input_dim_c, critic_hidden_dims, activation,
                            [128, 64], [128, 64], priv_encoder_dims[1], num_hist, num_prop)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        # Process observations
        prop_obs = observations[:, :48]  # First 48 dimensions are proprioceptive
        priv_obs = observations[:, 48:]  # Rest are privileged
        
        # Encode history and privileged information
        hist_latent = self.history_encoder(prop_obs.reshape(-1, 10, 48))  # 10 timesteps
        priv_latent = self.priv_encoder(priv_obs)
        
        # Combine all features
        combined_obs = torch.cat([observations, hist_latent, priv_latent], dim=-1)
        
        # Get actions and adaptive gains
        actions, adaptive_gains = self.actor(combined_obs, hist_encoding)
        
        # Create distribution
        self.distribution = Normal(actions, self.actor.log_std.exp())
        
        return adaptive_gains

    def act(self, observations, critic_observations, hidden_states=None, masks=None):
        adaptive_gains = self.update_distribution(observations, True)
        actions = self.distribution.sample()
        actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        
        # Evaluate critic
        values = self.evaluate(critic_observations, hidden_states, masks)
        
        return actions, actions_log_prob, values, adaptive_gains

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False):
        adaptive_gains = self.update_distribution(observations, hist_encoding)
        return self.distribution.mean, adaptive_gains

    def evaluate(self, critic_observations, hidden_states=None, masks=None):
        # Process critic observations similar to actor
        prop_obs = critic_observations[:, :48]
        priv_obs = critic_observations[:, 48:]
        
        hist_latent = self.history_encoder(prop_obs.reshape(-1, 10, 48))
        priv_latent = self.priv_encoder(priv_obs)
        
        combined_obs = torch.cat([critic_observations, hist_latent, priv_latent], dim=-1)
        
        return self.critic(combined_obs)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None 