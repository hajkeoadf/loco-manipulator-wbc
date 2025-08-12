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
from tqdm import tqdm

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
        self.dagger_update_freq = self.alg_cfg['dagger_update_freq']

        # 获取观测维度
        num_actor_obs = self.env.num_obs + self.env.get_num_priv()
        num_hist_obs = self.env.num_obs * self.env.get_num_hist_config()
        num_critic_obs = self.env.num_critic_obs if hasattr(self.env, 'num_critic_obs') else self.env.num_obs + self.env.get_num_priv()
        num_actions = self.env.num_actions

        # 创建Actor-Critic网络
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs=env.get_num_observations(),
            num_critic_obs=env.get_num_critic_observations(),
            num_actions=env.get_num_actions_total(),
            actor_hidden_dims=env.get_actor_hidden_dims(),
            critic_hidden_dims=env.get_critic_hidden_dims(),
            priv_encoder_dims=env.get_priv_encoder_dims(),
            activation=env.get_activation(),
            init_std=env.get_init_std(),
            num_hist=env.get_obs_history_length(),
            num_priv=env.get_num_priv(),
            num_prop=env.get_num_prop(),
            num_leg_actions=env.get_num_leg_actions(),
            num_arm_actions=env.get_num_arm_actions(),
            adaptive_arm_gains=env.get_adaptive_arm_gains(),
            adaptive_arm_gains_scale=env.get_adaptive_arm_gains_scale(),
            leg_control_head_hidden_dims=env.get_leg_control_head_hidden_dims(),
            arm_control_head_hidden_dims=env.get_arm_control_head_hidden_dims(),
            device=self.device
        ).to(self.device)

        # 创建RSL PPO算法
        alg_class = eval(self.cfg["algorithm_class_name"])  # RSLPPO
        self.alg = alg_class(
            actor_critic=actor_critic,
            value_loss_coef=self.alg_cfg['value_loss_coef'],  
            use_clipped_value_loss=self.alg_cfg['use_clipped_value_loss'],
            clip_param=self.alg_cfg['clip_param'],
            entropy_coef=self.alg_cfg['entropy_coef'],
            num_learning_epochs=self.alg_cfg['num_learning_epochs'],
            num_mini_batches=self.alg_cfg['num_mini_batches'],
            learning_rate=self.alg_cfg['learning_rate'],
            schedule=self.alg_cfg['schedule'],
            gamma=self.alg_cfg['gamma'],
            lam=self.alg_cfg['lam'],
            desired_kl=self.alg_cfg['desired_kl'],
            max_grad_norm=self.alg_cfg['max_grad_norm'],
            mixing_schedule=self.alg_cfg['mixing_schedule'],
            priv_reg_coef_schedule=self.alg_cfg['priv_reg_coef_schedual'],
            dagger_update_freq=self.alg_cfg['dagger_update_freq'],
            torque_supervision=False,
            device=self.device
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        print(f"num_envs: {self.env.num_envs}")
        print(f"num_steps_per_env: {self.num_steps_per_env}")
        # 初始化存储
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_hist_obs],
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
        mean_priv_reg_loss = 0. 
        mean_hist_latent_loss = 0.

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
        # 此处，事实上obs, critic_obs一模一样
        obs, critic_obs, obs_history = self.env.get_observations()
        obs, critic_obs, obs_history = (
            obs.to(self.device),
            critic_obs.to(self.device),
            obs_history.to(self.device),
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
        for it in tqdm(range(self.current_learning_iteration, tot_iter), desc="Learning iterations"):
            self.env.update_command_curriculum()

            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in tqdm(range(self.num_steps_per_env), desc=f"Rollout (iter {it})", leave=False):
                    # 获取动作
                    actions, actions_log_prob, values, adaptive_gains = self.alg.act(obs, obs_history, critic_obs, hist_encoding)
                    
                    # 环境步进
                    (obs, leg_rewards, arm_rewards, dones, infos, obs_history, critic_obs) = self.env.step(actions)
                    
                    obs, obs_history, critic_obs, leg_rewards, arm_rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        critic_obs.to(self.device),
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

            if hist_encoding:
                mean_hist_latent_loss = self.alg.update_dagger()
            else:
                mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, mean_torque_supervision_loss, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            
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
        self.writer.add_scalar(
            "Loss/priv_reg_loss", locs["mean_priv_reg_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/mean_hist_latent_loss", locs["mean_hist_latent_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        
        # 记录课程学习参数
        # self.writer.add_scalar("Curriculum/value_mixing_ratio", locs["value_mixing_ratio"], locs["it"])
        # self.writer.add_scalar("Curriculum/torque_supervision_weight", locs["torque_supervision_weight"], locs["it"])
        
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

        if len(locs["rewbuffer"]) > 1:
            # 有足够数据计算标准差
            log_string = (
                " " * len(str)
                + f" \033[1m Leg Reward: {statistics.mean(locs['rewbuffer']):.2f} ± {statistics.stdev(locs['rewbuffer']):.2f} \033[0m "
                + f" \033[1m Arm Reward: {statistics.mean(locs['armrewbuffer']):.2f} ± {statistics.stdev(locs['armrewbuffer']):.2f} \033[0m "
                + f" \033[1m Episode Length: {statistics.mean(locs['lenbuffer']):.1f} ± {statistics.stdev(locs['lenbuffer']):.1f} \033[0m "
            )
        elif len(locs["rewbuffer"]) == 1:
            # 只有一个数据点，只显示均值
            log_string = (
                " " * len(str)
                + f" \033[1m Leg Reward: {statistics.mean(locs['rewbuffer']):.2f} (single data) \033[0m "
                + f" \033[1m Arm Reward: {statistics.mean(locs['armrewbuffer']):.2f} (single data) \033[0m "
                + f" \033[1m Episode Length: {statistics.mean(locs['lenbuffer']):.1f} (single data) \033[0m "
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