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
import numpy as np
import os

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from typing import Tuple, Dict

from legged_gym.envs.solefoot_flat.solefoot_flat import BipedSF
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain, Terrain_Perlin
from legged_gym.utils.math import *
from .solefoot_flat_with_arm_config import BipedCfgSFWithArm

import torch

def cart2sphere(cart):
    """Convert cartesian coordinates to spherical coordinates"""
    x, y, z = cart[..., 0], cart[..., 1], cart[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(torch.clamp(z / r, -1, 1))
    phi = torch.atan2(y, x)
    return torch.stack([r, theta, phi], dim=-1)

def sphere2cart(sphere):
    """Convert spherical coordinates to cartesian coordinates"""
    r, theta, phi = sphere[..., 0], sphere[..., 1], sphere[..., 2]
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def orientation_error(desired, current):
    """Compute orientation error between desired and current quaternions"""
    q1 = current / torch.norm(current, dim=-1, keepdim=True)
    q2 = desired / torch.norm(desired, dim=-1, keepdim=True)
    
    error = torch.zeros_like(q1)
    error[:, 0] = q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0] - q1[:, 2] * q2[:, 3] + q1[:, 3] * q2[:, 2]
    error[:, 1] = q1[:, 0] * q2[:, 2] + q1[:, 1] * q2[:, 3] - q1[:, 2] * q2[:, 0] - q1[:, 3] * q2[:, 1]
    error[:, 2] = q1[:, 0] * q2[:, 3] - q1[:, 1] * q2[:, 2] + q1[:, 2] * q2[:, 1] - q1[:, 3] * q2[:, 0]
    error[:, 3] = q1[:, 0] * q2[:, 0] + q1[:, 1] * q2[:, 1] + q1[:, 2] * q2[:, 2] + q1[:, 3] * q2[:, 3]
    
    return error

class BipedSFWithArm(BipedSF):
    cfg: BipedCfgSFWithArm

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._parse_cfg(self.cfg)
        self._init_arm_variables()

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations."""
        total_obs_dim = self.cfg.env.num_observations + self.cfg.env.num_privileged_obs

        # Add height measurements if enabled
        if self.cfg.terrain.measure_heights:
            total_obs_dim += self.cfg.env.num_height_samples
            
        noise_vec = torch.zeros(total_obs_dim, device=self.device, dtype=torch.float)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # Proprio Obs
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:20] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 14 DOFs
        noise_vec[20:34] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 14 DOFs
        noise_vec[34:39] = 0.0  # commands (5)
        noise_vec[39:53] = 0.0  # actions (14 DOFs)
        noise_vec[53:59] = 0.0  # sin, cos and gaits
        noise_vec[59:65] = 0.0  # arm goal position and orientation

        # Priv Obs
        noise_vec[self.num_obs:self.num_obs+self.cfg.env.num_privileged_obs] = 0.0
        
        # Height measurements if enabled
        if self.cfg.terrain.measure_heights:
            noise_vec[self.num_obs+self.cfg.env.num_privileged_obs:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        
        return noise_vec

    def _get_arm_dog_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations."""
        # Add height measurements if enabled
        if self.cfg.terrain.measure_heights:
            total_obs_dim += self.cfg.env.num_height_samples

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        dog_noise_vec = torch.zeros(self.cfg.env.dog_num_observations, device=self.device, dtype=torch.float)
        # Proprio Obs
        dog_noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        dog_noise_vec[3:6] = noise_scales.gravity * noise_level
        dog_noise_vec[6:14] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 8 DOFs
        dog_noise_vec[14:22] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 8 DOFs
        dog_noise_vec[22:27] = 0.0  # commands (5)
        dog_noise_vec[27:35] = 0.0  # actions (8 DOFs)
        dog_noise_vec[35:41] = 0.0  # sin, cos and gaits
        # Priv Obs

        arm_noise_vec = torch.zeros(self.cfg.env.arm_num_observations, device=self.device, dtype=torch.float)
        # Proprio Obs
        arm_noise_vec[0:6] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 6 DOFs
        arm_noise_vec[6:12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 6 DOFs
        arm_noise_vec[12:18] = 0.0  # actions (6 DOFs)
        arm_noise_vec[18:24] = 0.0  # sin, cos and gaits
        # Priv Obs

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        
        # 重新初始化command_ranges为张量（因为父类的_parse_cfg会将其重置为字典）
        if hasattr(self, 'num_envs'):
            self.command_ranges["lin_vel_x"] = torch.tensor(
                self.cfg.commands.ranges.lin_vel_x, device=self.device, dtype=torch.float
            ).repeat(self.num_envs, 1)
            self.command_ranges["lin_vel_y"] = torch.tensor(
                self.cfg.commands.ranges.lin_vel_y, device=self.device, dtype=torch.float
            ).repeat(self.num_envs, 1)
            self.command_ranges["ang_vel_yaw"] = torch.tensor(
                self.cfg.commands.ranges.ang_vel_yaw, device=self.device, dtype=torch.float
            ).repeat(self.num_envs, 1)
            self.command_ranges["base_height"] = torch.tensor(
                self.cfg.commands.ranges.base_height, device=self.device, dtype=torch.float
            ).repeat(self.num_envs, 1)
            self.command_ranges["stand_still"] = torch.tensor(
                self.cfg.commands.ranges.stand_still, device=self.device, dtype=torch.float
            ).repeat(self.num_envs, 1)
        
        # 机械臂相关配置
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
        self.init_goal_ee_l_ranges = self.goal_ee_l_ranges = np.array(self.goal_ee_ranges['init_pos_l'])
        self.init_goal_ee_p_ranges = self.goal_ee_p_ranges = np.array(self.goal_ee_ranges['init_pos_p'])
        self.init_goal_ee_y_ranges = self.goal_ee_y_ranges = np.array(self.goal_ee_ranges['init_pos_y'])
        self.final_goal_ee_l_ranges = np.array(self.goal_ee_ranges['final_pos_l'])
        self.final_goal_ee_p_ranges = np.array(self.goal_ee_ranges['final_pos_p'])
        self.final_goal_ee_y_ranges = np.array(self.goal_ee_ranges['final_pos_y'])
        self.final_tracking_ee_reward = self.cfg.goal_ee.ranges.final_tracking_ee_reward
        self.goal_ee_l_schedule = self.cfg.goal_ee.l_schedule
        self.goal_ee_p_schedule = self.cfg.goal_ee.p_schedule
        self.goal_ee_y_schedule = self.cfg.goal_ee.y_schedule
        self.tracking_ee_reward_schedule = self.cfg.goal_ee.tracking_ee_reward_schedule

        self.goal_ee_delta_orn_ranges = torch.tensor(self.goal_ee_ranges['final_delta_orn'])
        
        # 机械臂控制参数
        self.arm_osc_kp = torch.tensor(self.cfg.arm.osc_kp, device=self.device, dtype=torch.float)
        self.arm_osc_kd = torch.tensor(self.cfg.arm.osc_kd, device=self.device, dtype=torch.float)
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)
        
        # 课程学习相关变量
        self.update_counter = 0

        self.action_delay = self.cfg.env.action_delay

    def _prepare_reward_function(self):
        """Prepares reward functions for both legs and arm."""
        super()._prepare_reward_function()
        
        # 确保episode_sums包含所有必要的键
        if "termination" not in self.episode_sums:
            self.episode_sums["termination"] = torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        
        # 定义机械臂专用奖励函数列表
        arm_reward_names = [
            'tracking_ee_sphere',
            'tracking_ee_orn',
            'arm_energy_abs_sum'
        ]
        
        # 只添加机械臂相关的奖励函数
        all_reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.arm_reward_scales = {}
        self.arm_reward_functions = []
        self.arm_reward_names = []
        
        for name in arm_reward_names:
            if name in all_reward_scales and all_reward_scales[name] != 0:
                self.arm_reward_scales[name] = all_reward_scales[name]
                self.arm_reward_names.append(name)
                reward_func_name = '_reward_' + name
                if hasattr(self, reward_func_name):
                    self.arm_reward_functions.append(getattr(self, reward_func_name))
                else:
                    print(f"⚠️  警告: 找不到奖励函数 {reward_func_name}")
        
        # 添加termination奖励（如果存在）
        if 'termination' in all_reward_scales and all_reward_scales['termination'] != 0:
            self.arm_reward_scales['termination'] = all_reward_scales['termination']

        # 验证奖励函数
        print(f"\n🏆 机械臂奖励函数 (共{len(self.arm_reward_names)}个):")
        for name in self.arm_reward_names:
            scale = self.arm_reward_scales.get(name, 0)
            print(f"  ✅ {name}: scale={scale}")
        
        if 'termination' in self.arm_reward_scales:
            print(f"  ✅ termination: scale={self.arm_reward_scales['termination']}")
            
        print("="*60)

    def _create_envs(self):
        super()._create_envs()
        # self._prepare_reward_function()
        
        # 获取机械臂末端执行器索引 - 修复：使用正确的链接名称
        self.ee_idx = self.body_names_to_idx.get("link6", self.cfg.env.ee_idx)

    def _init_arm_variables(self):
        """Initialize arm-related variables."""
        # 目标生成相关
        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze() / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze() / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        
        # 末端执行器目标
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_start_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_delta_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart_base = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere_base = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart_base = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere_base = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 碰撞检测
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        
        # 命令模式
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])
        if self.cfg.goal_ee.command_mode == 'cart':
            self.curr_ee_goal = self.curr_ee_goal_cart
        else:
            self.curr_ee_goal = self.curr_ee_goal_sphere
            
        # 误差缩放
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        
        # 机械臂基座位置
        self.arm_base_overhead = torch.tensor([0., 0., 0.165], device=self.device)
        self.z_invariant_offset = torch.tensor([0.8], device=self.device).repeat(self.num_envs, 1)
        
        # 初始化目标
        self._get_init_start_ee_sphere()

    def _get_curriculum_value(self, schedule, init_range, final_range, counter):
        """Get curriculum value based on schedule and counter."""
        return np.clip((counter - schedule[0]) / (schedule[1] - schedule[0]), 0, 1) * (final_range - init_range) + init_range
    
    def update_command_curriculum(self):
        """Update curriculum values for arm commands and rewards."""
        self.update_counter += 1

        # 更新末端执行器目标范围
        self.goal_ee_l_ranges = self._get_curriculum_value(
            self.goal_ee_l_schedule, 
            self.init_goal_ee_l_ranges, 
            self.final_goal_ee_l_ranges, 
            self.update_counter
        )
        self.goal_ee_p_ranges = self._get_curriculum_value(
            self.goal_ee_p_schedule, 
            self.init_goal_ee_p_ranges, 
            self.final_goal_ee_p_ranges, 
            self.update_counter
        )
        self.goal_ee_y_ranges = self._get_curriculum_value(
            self.goal_ee_y_schedule, 
            self.init_goal_ee_y_ranges, 
            self.final_goal_ee_y_ranges, 
            self.update_counter
        )
        
        # 更新跟踪奖励权重
        if 'tracking_ee_sphere' in self.arm_reward_scales:
            self.arm_reward_scales['tracking_ee_sphere'] = self._get_curriculum_value(
                self.tracking_ee_reward_schedule, 
                0, 
                self.final_tracking_ee_reward, 
                self.update_counter
            )
        if 'tracking_ee_cart' in self.arm_reward_scales:
            self.arm_reward_scales['tracking_ee_cart'] = self._get_curriculum_value(
                self.tracking_ee_reward_schedule, 
                0, 
                self.final_tracking_ee_reward, 
                self.update_counter
            )

    def _init_buffers(self):
        """Initialize buffers including arm-related ones."""
        super()._init_buffers()
        
        # 机械臂相关缓冲区
        self.arm_rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 末端执行器信息 - 需要在post_physics_step中更新
        self.ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_orn = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_vel = torch.zeros(self.num_envs, 6, device=self.device)
        
        # 雅可比矩阵和质量矩阵 - 需要在需要时更新
        self.ee_j_eef = torch.zeros(self.num_envs, 6, 6, device=self.device)
        self.mm = torch.zeros(self.num_envs, 6, 6, device=self.device)
        
        # 期望姿态
        self.ee_orn_des = torch.tensor([0, 0.7071068, 0, 0.7071068], device=self.device).repeat((self.num_envs, 1))

        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.obs_history_length, self.cfg.env.num_observations, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)

    def _get_init_start_ee_sphere(self):
        """Initialize starting end-effector position in spherical coordinates."""
        init_start_ee_cart = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        init_start_ee_cart = self.rigid_body_state[:, self.ee_idx, :3]
        self.ee_start_cart = init_start_ee_cart.clone()
        self.init_start_ee_sphere = cart2sphere(init_start_ee_cart)

    def _resample_ee_goal_sphere_once(self, env_ids):
        """Resample end-effector goal in cartesian coordinates and convert to spherical."""
        # 从配置中读取笛卡尔坐标范围
        init_x_range = np.array(self.goal_ee_ranges['init_pos_x'])
        init_y_range = np.array(self.goal_ee_ranges['init_pos_y'])
        init_z_range = np.array(self.goal_ee_ranges['init_pos_z'])
        
        final_x_range = np.array(self.goal_ee_ranges['final_pos_x'])
        final_y_range = np.array(self.goal_ee_ranges['final_pos_y'])
        final_z_range = np.array(self.goal_ee_ranges['final_pos_z'])
        
        # 根据课程学习进度插值计算当前范围
        progress = np.clip(self.update_counter / 5000, 0, 1)  # 假设5000步完成课程学习
        
        current_x_range = init_x_range + progress * (final_x_range - init_x_range)
        current_y_range = init_y_range + progress * (final_y_range - init_y_range)
        current_z_range = init_z_range + progress * (final_z_range - init_z_range)
        
        # 生成笛卡尔坐标目标（相对坐标）
        ee_goal_cart = torch.zeros(len(env_ids), 3, device=self.device)
        ee_goal_cart[:, 0] = torch_rand_float(current_x_range[0], current_x_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        ee_goal_cart[:, 1] = torch_rand_float(current_y_range[0], current_y_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        ee_goal_cart[:, 2] = torch_rand_float(current_z_range[0], current_z_range[1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.ee_goal_cart_base[env_ids] = ee_goal_cart
        self.ee_goal_sphere_base[env_ids] = cart2sphere(ee_goal_cart)

        # 转换为世界坐标系
        ee_goal_cart_world = self.root_states[env_ids, :3] + quat_apply(self.base_quat[env_ids], ee_goal_cart)
        
        # 转换为球坐标
        self.ee_goal_cart[env_ids] = ee_goal_cart_world
        self.ee_goal_sphere[env_ids] = cart2sphere(ee_goal_cart_world)

    def _resample_ee_goal_orn_once(self, env_ids):
        """Resample end-effector orientation goal."""
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_delta_orn_ranges[0, 0], self.goal_ee_delta_orn_ranges[0, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_delta_orn_ranges[1, 0], self.goal_ee_delta_orn_ranges[1, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_delta_orn_ranges[2, 0], self.goal_ee_delta_orn_ranges[2, 1], (len(env_ids), 1), device=self.device)
        self.ee_goal_delta_orn_euler[env_ids] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)
        self.ee_goal_orn_euler[env_ids] = torch_wrap_to_pi_minuspi(self.ee_goal_delta_orn_euler[env_ids] + self.base_yaw_euler[env_ids])

    def _resample_ee_goal(self, env_ids, is_init=False):
        """Resample end-effector goals with collision checking."""
        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            self._resample_ee_goal_orn_once(env_ids)
            if is_init:
                self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[env_ids].clone()
            else:
                self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
            for i in range(10):
                self._resample_ee_goal_sphere_once(env_ids)
                collision_mask = self.collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break
                    
            self.goal_timer[init_env_ids] = 0.0

    def collision_check(self, env_ids):
        """Check for collisions along the trajectory."""
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ..., None], self.collision_check_t).squeeze(-1)
        ee_target_cart = sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def update_curr_ee_goal(self):
        """Update current end-effector goal based on trajectory."""
        # t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        # self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        # self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        self.curr_ee_goal_sphere[:] = self.ee_goal_sphere
        self.curr_ee_goal_cart[:] = self.ee_goal_cart
        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        if len(resample_id) > 0:
            print(f"resample_id: {resample_id}")
        self._resample_ee_goal(resample_id)

    def get_arm_ee_control_torques(self):
        """Compute operational space control torques for the arm."""
        # 更新雅可比矩阵和质量矩阵
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        
        # 计算逆质量矩阵
        m_inv = torch.pinverse(self.mm)
        m_eef = torch.pinverse(self.ee_j_eef @ m_inv @ torch.transpose(self.ee_j_eef, 1, 2))
        
        # 计算姿态误差
        ee_orn_normalized = self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1)
        orn_err = orientation_error(self.ee_orn_des, ee_orn_normalized)
        
        # 计算位置误差
        # pos_err = (torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + 
        #           quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart) - self.ee_pos)
        pos_err = self.curr_ee_goal_cart - self.ee_pos
        
        # 组合误差
        dpose = torch.cat([pos_err, orn_err], -1)
        
        # 计算控制输出
        u = (torch.transpose(self.ee_j_eef, 1, 2) @ m_eef @ 
             (self.arm_osc_kp * dpose - self.arm_osc_kd * self.ee_vel)[:, :6].unsqueeze(-1)).squeeze(-1)
        
        return u

    def _compute_torques(self, actions):
        """Compute torques for both legs and arm."""
        # 使用父类方法计算所有关节的PD控制扭矩
        pd_torques = super()._compute_torques(actions)
        
        # 根据配置决定是否使用操作空间控制
        if hasattr(self.cfg.control, 'torque_supervision') and self.cfg.control.torque_supervision:
            # 获取操作空间控制扭矩
            try:
                arm_osc_torques = self.get_arm_ee_control_torques()
                # 将操作空间控制扭矩应用到机械臂关节（8-13）
                pd_torques[:, 8:14] += arm_osc_torques
            except Exception as e:
                print(f"⚠️  操作空间控制计算失败: {e}")
                # 如果操作空间控制失败，继续使用PD控制
                pass
        
        return pd_torques

    def _compute_curr_ee_goal_in_base_frame(self, flag=False):
        """实时计算当前EE goal在base坐标系下的坐标
        
        Returns:
            curr_ee_goal_cart_base: 当前时刻EE goal在base坐标系下的笛卡尔坐标 [num_envs, 3]
        """
        # 获取机器人基座当前位置和朝向
        base_pos = self.root_states[:, :3]  # 基座当前世界位置
        base_quat = self.root_states[:, 3:7]  # 基座当前世界朝向
        
        if flag:
            print(f"\nbase_pos: {base_pos[0]}")
            print(f"base_quat: {base_quat[0]}\n")

        # 计算EE goal相对于基座的位置（世界坐标系下的相对位置）
        ee_goal_relative_world = self.curr_ee_goal_cart - base_pos
        
        # 将相对位置从世界坐标系转换到base坐标系
        # 使用基座四元数的逆旋转
        base_quat_inv = quat_conjugate(base_quat)
        curr_ee_goal_cart_base = quat_apply(base_quat_inv, ee_goal_relative_world)
        
        return curr_ee_goal_cart_base

    def _compute_curr_ee_goal_sphere_in_base_frame(self):
        """实时计算当前EE goal在base坐标系下的球坐标
        
        Returns:
            curr_ee_goal_sphere_base: 当前时刻EE goal在base坐标系下的球坐标 [num_envs, 3]
        """
        curr_ee_goal_cart_base = self._compute_curr_ee_goal_in_base_frame()
        return cart2sphere(curr_ee_goal_cart_base)

    def compute_observations(self):
        """Compute observations including arm-related ones."""
        # 获取基础观察
        obs_buf, _, arm_obs_buf, dog_obs_buf = super().compute_self_observations()

        curr_ee_goal_sphere_base = self._compute_curr_ee_goal_sphere_in_base_frame()
        
        # 添加机械臂相关观察
        arm_obs = torch.cat([
            curr_ee_goal_sphere_base,  # 当前目标位置 (3)
            self.ee_goal_delta_orn_euler  # 目标姿态 (3)
        ], dim=-1)

        # 组合观察
        obs_buf = torch.cat([obs_buf, arm_obs], dim=-1)

        # 加入priv_obs
        if self.cfg.env.observe_priv:
            # 检查 friction_coeffs_tensor 的维度
            if len(self.friction_coeffs_tensor.shape) == 1:
                # 如果是一维，扩展为 [batch_size, 1] 的形状
                friction_coeffs_expanded = self.friction_coeffs_tensor.unsqueeze(-1)
                priv_buf = torch.cat((
                    self.mass_params_tensor,
                    friction_coeffs_expanded
                ), dim=-1)
            else:
                # 如果已经是二维，直接使用原来的
                priv_buf = torch.cat((
                    self.mass_params_tensor,
                    self.friction_coeffs_tensor
                ), dim=-1)
            obs_buf = torch.cat([obs_buf, priv_buf], dim=-1)

        # 加入高度观测（如果有）
        if self.cfg.terrain.measure_heights:   
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            obs_buf = torch.cat((obs_buf, heights), dim=-1)

        # 加噪声
        if self.add_noise:
            noise_scale_vec = self._get_noise_scale_vec(self.cfg)
            noise = (2 * torch.rand_like(obs_buf) - 1) * noise_scale_vec
            obs_buf = obs_buf + noise

        self.obs_buf = obs_buf
        self.critic_obs_buf = obs_buf   

        # arm_noise, dog_noise = self._get_arm_dog_noise_scale_vec(self.cfg)
        self.arm_obs_buf = torch.cat([arm_obs_buf, arm_obs], dim=-1)
        self.dog_obs_buf = dog_obs_buf

        # 更新历史观测缓冲区
        curr_hist_obs = obs_buf[:, :self.num_obs]
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([curr_hist_obs] * self.cfg.env.obs_history_length, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                curr_hist_obs.unsqueeze(1)
            ], dim=1)
        )

        return self.obs_buf

    def get_arm_observations(self):
        return {
            "obs": self.arm_obs_buf,
            "privileged_obs": self.arm_privileged_obs_buf,
            "obs_history": self.arm_obs_history_buf
        }
    
    def get_dog_observations(self):
        return {
            "obs": self.dog_obs_buf,
            "privileged_obs": self.dog_privileged_obs_buf,
            "obs_history": self.dog_obs_history_buf
        }

    def compute_reward(self):
        """Compute rewards for the arm."""
        # 计算腿部奖励
        super().compute_reward()
        
        # 计算机械臂奖励
        self.arm_rew_buf[:] = 0.
        for i in range(len(self.arm_reward_functions)):
            name = self.arm_reward_names[i]
            rew = self.arm_reward_functions[i]() * self.arm_reward_scales[name]
            self.arm_rew_buf += rew
            self.episode_sums[name] += rew
            
        if self.cfg.rewards.only_positive_rewards:
            self.arm_rew_buf[:] = torch.clip(self.arm_rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.arm_reward_scales:
            rew = self._reward_termination() * self.arm_reward_scales["termination"]
            self.arm_rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reward_tracking_ee_sphere(self):
        """Reward for tracking end-effector position in spherical coordinates."""
        # ee_pos_local = quat_rotate_inverse(self.base_yaw_quat, 
        #                                  self.ee_pos - torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1))
        # ee_pos_error = torch.sum(torch.abs(cart2sphere(ee_pos_local) - self.curr_ee_goal_sphere) * self.sphere_error_scale, dim=1)
        ee_pos_error = torch.sum(torch.abs(self.ee_pos - self.curr_ee_goal_sphere) * self.sphere_error_scale, dim=1)
        return torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma)

    def _reward_tracking_ee_cart(self):
        """Reward for tracking end-effector position in cartesian coordinates."""
        target_ee = self.curr_ee_goal_cart
        ee_pos_error = torch.sum(torch.abs(self.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma)

    def _reward_tracking_ee_orn(self):
        """Reward for tracking end-effector orientation."""
        ee_orn_euler = torch.stack(euler_from_quat(self.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.ee_goal_orn_euler - ee_orn_euler)) * self.orn_error_scale, dim=1)
        return torch.exp(-orn_err/self.cfg.rewards.tracking_ee_sigma)

    def _reward_arm_energy_abs_sum(self):
        """Reward for arm energy consumption."""
        return torch.sum(torch.abs(self.torques[:, 8:14] * self.dof_vel[:, 8:14]), dim=1)

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)

        self.base_lin_acc = (self.base_lin_vel - self.last_base_lin_vel) / self.dt
        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, self.base_lin_acc)

        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.dof_pos_int += (self.dof_pos - self.raw_default_dof_pos) * self.dt
        self.power = torch.abs(self.torques * self.dof_vel)

        # self.dof_jerk = (self.last_dof_acc - self.dof_acc) / self.dt

        self.compute_foot_state()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_observations()
        self.compute_reward()

        self._post_physics_step_callback()

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # self.last_dof_acc[:] = self.dof_acc[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_foot_positions[:] = self.foot_positions[:]
        
        # 更新末端执行器状态
        self.ee_pos = self.rigid_body_state[:, self.ee_idx, :3]
        self.ee_orn = self.rigid_body_state[:, self.ee_idx, 3:7]
        self.ee_vel = self.rigid_body_state[:, self.ee_idx, 7:]

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # 更新末端执行器目标
        self.update_curr_ee_goal()

        if self.viewer:
            self.gym.clear_lines(self.viewer)
            self._draw_debug_vis()
            # self._draw_ee_goal()

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum()

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._check_walk_stability(env_ids)
        self._resample_commands(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.goal_timer[env_ids] = 0.

        # 先计算高度测量
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # 使用完整的观测（包含高度测量）
        self.compute_observations()
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.dof_pos_int[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
            
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def step(self, actions):
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.arm_rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history_buf,
            self.critic_obs_buf
        )

    def _draw_debug_vis(self):
        """Draw debug visualizations for arm with detailed target analysis."""
        # 绘制末端执行器目标位置
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
        transformed_target_ee = self.curr_ee_goal_cart

        # 绘制当前末端执行器位置
        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        ee_pose = self.rigid_body_state[:, self.ee_idx, :3]

        # 绘制base位置
        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 1, 1))
        base_pose = self.root_states[:, :3]

        self._draw_ee_orientation_vis()
        # self._debug_coordinate_consistency()
        
        for i in range(self.num_envs):
            # 目标位置（黄色球）
            sphere_pose = gymapi.Transform(gymapi.Vec3(transformed_target_ee[i, 0], transformed_target_ee[i, 1], transformed_target_ee[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            
            # 当前位置（蓝色球）
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2)

            # 绘制base位置
            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(base_pose[i, 0], base_pose[i, 1], base_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3)

    def _draw_ee_goal(self):
        """Draw end-effector goal trajectory."""
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))
        
        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze()
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        
        for i in range(10):
            ee_target_cart = sphere2cart(ee_target_all_sphere[..., i])
            ee_target_all_cart_world[..., i] = ee_target_cart
        
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def get_observations(self):
        """Get observations for RSL algorithm compatibility."""
        # # 确保观察缓冲区已更新
        # if self.obs_buf is None or self.obs_buf.shape[1] != self.num_obs:
        #     self.compute_observations()
        
        return (
            self.obs_buf,
            self.critic_obs_buf,
            self.obs_history_buf
        )

    def get_arm_observations(self):
        """Get arm observations for RD algorithm compatibility."""
        

    def get_privileged_observations(self):
        """Get privileged observations for RSL algorithm compatibility."""
        if self.privileged_obs_buf is not None:
            return self.privileged_obs_buf
        else:
            return self.obs_buf

    def get_rewards(self):
        """Get rewards for RSL algorithm compatibility."""
        return self.rew_buf

    def get_arm_rewards(self):
        """Get arm-specific rewards."""
        return self.arm_rew_buf

    def get_dones(self):
        """Get done flags for RSL algorithm compatibility."""
        return self.reset_buf

    def get_extras(self):
        """Get extra information for RSL algorithm compatibility."""
        return self.extras

    def get_num_leg_actions(self):
        """Get number of leg actions for RSL algorithm."""
        return 8  # 8个腿部关节

    def get_num_arm_actions(self):
        """Get number of arm actions for RSL algorithm."""
        return 6  # 6个机械臂关节

    def get_num_actions(self):
        """Get total number of actions."""
        return self.num_actions

    def get_num_observations(self):
        """Get number of observations."""
        return self.num_obs

    def get_num_critic_observations(self):
        """Get number of critic observations."""
        return self.num_critic_obs

    def get_num_privileged_obs(self):
        """Get number of privileged observations."""
        return self.num_privileged_obs

    def get_obs_history_length(self):
        """Get observation history length."""
        return self.cfg.env.obs_history_length

    def get_num_proprio(self):
        """Get number of proprioceptive observations."""
        return self.cfg.env.num_observations

    def get_num_height_samples(self):
        """Get number of height samples."""
        return self.cfg.env.num_height_samples

    def get_action_delay(self):
        """Get action delay."""
        return self.cfg.env.action_delay

    def get_adaptive_arm_gains(self):
        """Get whether adaptive arm gains are enabled."""
        return self.cfg.control.adaptive_arm_gains

    def get_adaptive_arm_gains_scale(self):
        """Get adaptive arm gains scale."""
        return 0.1  # 默认值，可以在配置中设置

    def get_leg_control_head_hidden_dims(self):
        """Get leg control head hidden dimensions."""
        return [128, 64]  # 默认值，可以在配置中设置

    def get_arm_control_head_hidden_dims(self):
        """Get arm control head hidden dimensions."""
        return [128, 64]  # 默认值，可以在配置中设置

    def get_priv_encoder_dims(self):
        """Get privileged encoder dimensions."""
        return [64, 20]  # 默认值，可以在配置中设置

    def get_num_hist(self):
        """Get number of history steps."""
        return self.cfg.env.obs_history_length

    def get_num_prop(self):
        """Get number of proprioceptive observations."""
        return self.cfg.env.num_observations

    def get_num_priv(self):
        """Get number of privileged observations."""
        return self.num_privileged_obs

    def get_num_actor_obs(self):
        """Get number of actor observations."""
        return self.num_obs

    def get_num_critic_obs(self):
        """Get number of critic observations."""
        return self.num_critic_obs

    def get_num_actions_total(self):
        """Get total number of actions."""
        return self.num_actions

    def get_activation(self):
        """Get activation function."""
        return "elu"  # 默认值，可以在配置中设置

    def get_init_std(self):
        """Get initial standard deviation."""
        return 1.0  # 默认值，可以在配置中设置

    def get_actor_hidden_dims(self):
        """Get actor hidden dimensions."""
        return [512, 256, 128]  # 默认值，可以在配置中设置

    def get_critic_hidden_dims(self):
        """Get critic hidden dimensions."""
        return [512, 256, 128]  # 默认值，可以在配置中设置

    def get_priv_encoder_dims_config(self):
        """Get privileged encoder dimensions from config."""
        return [64, 20]  # 默认值，可以在配置中设置

    def get_num_hist_config(self):
        """Get number of history steps from config."""
        return self.cfg.env.obs_history_length

    def get_num_prop_config(self):
        """Get number of proprioceptive observations from config."""
        return self.cfg.env.num_observations

    def get_num_priv_config(self):
        """Get number of privileged observations from config."""
        return self.num_privileged_obs - self.cfg.env.num_observations

    def get_num_actor_obs_config(self):
        """Get number of actor observations from config."""
        return self.num_obs

    def get_num_critic_obs_config(self):
        """Get number of critic observations from config."""
        return self.num_critic_obs

    def get_num_actions_total_config(self):
        """Get total number of actions from config."""
        return self.num_actions

    def get_activation_config(self):
        """Get activation function from config."""
        return "elu"  # 默认值，可以在配置中设置

    def get_init_std_config(self):
        """Get initial standard deviation from config."""
        return 1.0  # 默认值，可以在配置中设置

    def get_actor_hidden_dims_config(self):
        """Get actor hidden dimensions from config."""
        return [512, 256, 128]  # 默认值，可以在配置中设置

    def get_critic_hidden_dims_config(self):
        """Get critic hidden dimensions from config."""
        return [512, 256, 128]  # 默认值，可以在配置中设置

    def get_leg_control_head_hidden_dims_config(self):
        """Get leg control head hidden dimensions from config."""
        return [128, 64]  # 默认值，可以在配置中设置

    def get_arm_control_head_hidden_dims_config(self):
        """Get arm control head hidden dimensions from config."""
        return [128, 64]  # 默认值，可以在配置中设置

    def get_priv_encoder_dims_config(self):
        """Get privileged encoder dimensions from config."""
        return [64, 20]  # 默认值，可以在配置中设置

    def get_adaptive_arm_gains_config(self):
        """Get adaptive arm gains from config."""
        return self.cfg.control.adaptive_arm_gains

    def get_adaptive_arm_gains_scale_config(self):
        """Get adaptive arm gains scale from config."""
        return self.cfg.control.adaptive_arm_gains_scale

    def update_curriculum(self):
        """Public method to update curriculum learning."""
        if self.cfg.commands.curriculum:
            self.update_command_curriculum()

    def _draw_init_start_sphere_vis(self):
        """Draw visualization for initial start sphere positions of each environment."""
        # 创建球体几何体 - 使用绿色表示初始起点
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(0, 1, 0))
        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(0, 0, 1))

        # ✅ 修复：直接使用ee_start_sphere而不是init_start_ee_sphere
        # 因为ee_start_sphere在reset时已经被正确设置为ee的实际位置
        if hasattr(self, 'ee_start_sphere'):
            # 将球坐标转换为笛卡尔坐标
            # start_sphere_cart = sphere2cart(self.ee_start_sphere)
            ee_goal_cart = self.root_states[:, :3] + quat_apply(self.base_yaw_quat, self.ee_goal_cart)
            
            # 绘制每个环境的初始起点
            for i in range(self.num_envs):
                # 获取该环境的初始起点位置
                # start_pos = start_sphere_cart[i]
                ee_goal_pos = ee_goal_cart[i]
                
                # 创建变换矩阵
                # pose = gymapi.Transform(
                #     gymapi.Vec3(start_pos[0], start_pos[1], start_pos[2]), 
                #     r=None
                # )
                pose_2 = gymapi.Transform(  
                    gymapi.Vec3(ee_goal_pos[0], ee_goal_pos[1], ee_goal_pos[2]), 
                    r=None
                )
                # 绘制球体
                # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)
                gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], pose_2)

    def _draw_ee_orientation_vis(self):
        """绘制EE朝向可视化"""
        if not hasattr(self, 'viewer') or self.viewer is None:
            return
            
        # 清除之前的线条
        self.gym.clear_lines(self.viewer)
        
        # 为每个环境绘制EE朝向
        for i in range(self.num_envs):
            # 获取当前EE位置和朝向
            current_ee_pos = self.ee_pos[i]
            current_ee_orn = self.ee_orn[i]
            
            # 获取目标EE位置和朝向
            target_ee_pos = self.curr_ee_goal_cart[i]
            target_ee_orn = self.ee_goal_orn_euler[i]  # 欧拉角格式
            
            # 将目标欧拉角转换为四元数
            target_ee_orn_quat = quat_from_euler_xyz(
                target_ee_orn[0],  # roll
                target_ee_orn[1],  # pitch  
                target_ee_orn[2]   # yaw
            )
            
            # 绘制当前EE朝向（蓝色）
            self._draw_single_ee_orientation(
                current_ee_pos, current_ee_orn, 
                color=(0, 0, 1),  # 蓝色
                scale=0.1,        # 坐标轴长度
                env_id=i
            )
            
            # 绘制目标EE朝向（红色）
            self._draw_single_ee_orientation(
                target_ee_pos, target_ee_orn_quat,
                color=(1, 0, 0),  # 红色
                scale=0.1,        # 坐标轴长度
                env_id=i
            )

    def _draw_single_ee_orientation(self, position, quaternion, color, scale, env_id):
        """绘制单个EE的朝向坐标轴"""
        # 创建坐标轴几何体
        axes_geom = gymutil.AxesGeometry(scale)
        
        # 创建变换矩阵
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(position[0], position[1], position[2])
        pose.r = gymapi.Quat(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        
        # 绘制坐标轴
        gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[env_id], pose)
        
            