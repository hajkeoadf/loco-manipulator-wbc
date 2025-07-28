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

import time
import os
from collections import deque
import statistics
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from .rsl_ppo import RSLPPO
from .rsl_actor_critic import ActorCritic
from legged_gym.envs.vec_env import VecEnv


class RSLOnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # 获取观测维度
        num_actor_obs = self.env.num_obs
        num_critic_obs = self.env.num_critic_obs if hasattr(self.env, 'num_critic_obs') else self.env.num_obs
        num_actions = self.env.num_actions

        # 创建Actor-Critic网络
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # 创建RSL PPO算法
        alg_class = eval(self.cfg["algorithm_class_name"])  # RSLPPO
        self.alg = alg_class(
            actor_critic=actor_critic,
            device=self.device,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # 初始化存储
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [num_actions],
        )

        # 设置机械臂默认参数（如果使用自适应增益）
        if hasattr(self.env, 'p_gains') and hasattr(self.env, 'd_gains') and hasattr(self.env, 'default_dof_pos'):
            # 假设机械臂关节是最后6个
            arm_p_gains = self.env.p_gains[-6:] if len(self.env.p_gains) >= 6 else torch.ones(6) * 100.0
            arm_d_gains = self.env.d_gains[-6:] if len(self.env.d_gains) >= 6 else torch.ones(6) * 10.0
            arm_dof_pos = self.env.default_dof_pos[-6:] if len(self.env.default_dof_pos) >= 6 else torch.zeros(6)
            
            self.alg.set_arm_default_coeffs(
                default_arm_p_gains=arm_p_gains,
                default_arm_d_gains=arm_d_gains,
                default_arm_dof_pos=arm_dof_pos
            )

        # 日志相关
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # 重置环境
        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # 初始化指标
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_entropy_loss = 0.
        mean_torque_supervision_loss = 0.
        value_mixing_ratio = 0.
        torque_supervision_weight = 0.

        # 初始化writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from ..utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 获取初始观测
        obs, obs_history, commands, critic_obs = self.env.get_observations()
        obs, obs_history, commands, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            commands.to(self.device),
            critic_obs.to(self.device),
        )

        self.alg.actor_critic.train()  # 切换到训练模式

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        armrewbuffer = deque(maxlen=100)  # 机械臂奖励buffer
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_arm_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # 获取动作
                    actions, actions_log_prob, values, adaptive_gains = self.alg.act(obs, critic_obs)
                    
                    # 环境步进
                    (obs, rewards, dones, infos, obs_history, commands, critic_obs_buf) = self.env.step(actions)
                    
                    # 分离腿部和机械臂奖励
                    if isinstance(rewards, torch.Tensor) and rewards.dim() == 2 and rewards.shape[1] == 2:
                        # 如果奖励已经是分离的 [leg_reward, arm_reward]
                        leg_rewards = rewards[:, 0:1]
                        arm_rewards = rewards[:, 1:2]
                    else:
                        # 如果奖励是统一的，需要分离
                        leg_rewards = rewards * 0.7  # 70%给腿部
                        arm_rewards = rewards * 0.3  # 30%给机械臂
                    
                    obs, obs_history, commands, critic_obs, leg_rewards, arm_rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        commands.to(self.device),
                        critic_obs_buf.to(self.device),
                        leg_rewards.to(self.device),
                        arm_rewards.to(self.device),
                        dones.to(self.device),
                    )
                    
                    # 处理环境步
                    self.alg.process_env_step(leg_rewards, arm_rewards, dones, infos)

                    if self.log_dir is not None:
                        # 记录信息
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += leg_rewards.squeeze(-1)
                        cur_arm_reward_sum += arm_rewards.squeeze(-1)
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        armrewbuffer.extend(
                            cur_arm_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_arm_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # 学习步骤
                start = stop
                self.alg.compute_returns(critic_obs)

            # 更新策略
            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_entropy_loss,
                mean_torque_supervision_loss,
            ) = self.alg.update()
            
            # 获取课程学习参数
            value_mixing_ratio = self.alg.get_value_mixing_ratio()
            torque_supervision_weight = self.alg.get_torque_supervision_weight()
            
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 处理标量和零维张量信息
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        # 计算FPS
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # 记录损失
        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/entropy", locs["mean_entropy_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/torque_supervision", locs["mean_torque_supervision_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        
        # 记录课程学习参数
        self.writer.add_scalar("Curriculum/value_mixing_ratio", locs["value_mixing_ratio"], locs["it"])
        self.writer.add_scalar("Curriculum/torque_supervision_weight", locs["torque_supervision_weight"], locs["it"])
        
        # 记录性能指标
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        
        # 记录奖励
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_leg_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            if self.logger_type != "wandb":
                self.writer.add_scalar(
                    "Train/mean_leg_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )
        
        # 记录机械臂奖励
        if len(locs["armrewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_arm_reward", statistics.mean(locs["armrewbuffer"]), locs["it"]
            )
            if self.logger_type != "wandb":
                self.writer.add_scalar(
                    "Train/mean_arm_reward/time",
                    statistics.mean(locs["armrewbuffer"]),
                    self.tot_time,
                )

        # 打印训练信息
        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                " " * len(str)
                + f" \033[1m Leg Reward: {statistics.mean(locs['rewbuffer']):.2f} ± {statistics.stdev(locs['rewbuffer']):.2f} \033[0m "
                + f" \033[1m Arm Reward: {statistics.mean(locs['armrewbuffer']):.2f} ± {statistics.stdev(locs['armrewbuffer']):.2f} \033[0m "
                + f" \033[1m Episode Length: {statistics.mean(locs['lenbuffer']):.1f} ± {statistics.stdev(locs['lenbuffer']):.1f} \033[0m "
            )
        else:
            log_string = " " * len(str) + " \033[1m No training data \033[0m "

        print(str)
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": {
                    "train_cfg": {
                        "runner": self.cfg,
                        "algorithm": self.alg_cfg,
                        "policy": self.policy_cfg,
                    },
                },
            },
            path,
        )

    def load(self, path, load_optimizer=False):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # 切换到评估模式
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

    def get_actor_critic(self, device=None):
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic 