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


from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from torch import Tensor

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float, CubicSpline
)
from legged_gym.utils.helpers import get_scale_shift
from .solefoot_flat import BipedSF
from .solefoot_flat_config import BipedCfgSF

import math
from time import time
from warnings import WarningMessage
import numpy as np
import os
from typing import Tuple, Dict
import random


class BipedSFArm(BipedSF):
    """继承自BipedSF的机械臂子类，专门处理双足+机械臂的机器人"""
    
    def __init__(
        self, cfg: BipedCfgSF, sim_params, physics_engine, sim_device, headless
    ):
        """初始化机械臂环境"""
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 机械臂相关参数
        self.num_leg_dofs = 8  # 双足关节数量
        self.num_arm_dofs = 6  # 机械臂关节数量
        self.num_dofs = self.num_leg_dofs + self.num_arm_dofs  # 总关节数量
        
        # 机械臂关节索引
        self.leg_dof_indices = torch.arange(0, self.num_leg_dofs, device=self.device)
        self.arm_dof_indices = torch.arange(self.num_leg_dofs, self.num_dofs, device=self.device)
        
        # 机械臂相关缓冲区
        self._init_arm_buffers()
        
        # 机械臂目标位置和姿态
        self.arm_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.arm_target_orn = torch.zeros(self.num_envs, 4, device=self.device)  # 四元数
        
        # 机械臂控制参数
        self.arm_p_gains = torch.tensor([100.0, 100.0, 100.0, 50.0, 50.0, 50.0], device=self.device)
        self.arm_d_gains = torch.tensor([10.0, 10.0, 10.0, 5.0, 5.0, 5.0], device=self.device)
        
        # 机械臂默认位置
        self.arm_default_pos = torch.zeros(self.num_arm_dofs, device=self.device)
        
        # 末端执行器索引
        self.ee_idx = self.cfg.env.ee_idx
        
    def _init_arm_buffers(self):
        """初始化机械臂相关的缓冲区"""
        # 机械臂关节状态
        self.arm_dof_pos = torch.zeros(self.num_envs, self.num_arm_dofs, device=self.device)
        self.arm_dof_vel = torch.zeros(self.num_envs, self.num_arm_dofs, device=self.device)
        self.arm_torques = torch.zeros(self.num_envs, self.num_arm_dofs, device=self.device)
        
        # 机械臂末端执行器状态
        self.arm_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.arm_ee_orn = torch.zeros(self.num_envs, 4, device=self.device)  # 四元数
        self.arm_ee_vel = torch.zeros(self.num_envs, 6, device=self.device)  # 线速度 + 角速度
        
        # 机械臂雅可比矩阵
        self.arm_jacobian = torch.zeros(self.num_envs, 6, self.num_arm_dofs, device=self.device)
        
        # 机械臂目标状态
        self.arm_target_dof_pos = torch.zeros(self.num_envs, self.num_arm_dofs, device=self.device)
        self.arm_target_dof_vel = torch.zeros(self.num_envs, self.num_arm_dofs, device=self.device)
        
    def get_leg_obs(self):
        """获取腿部观测"""
        # 腿部关节位置、速度、力矩
        leg_pos = self.dof_pos[:, :self.num_leg_dofs]
        leg_vel = self.dof_vel[:, :self.num_leg_dofs]
        leg_torque = self.torques[:, :self.num_leg_dofs] if hasattr(self, "torques") else torch.zeros_like(leg_pos)
        
        # 足端状态
        foot_pos = self.foot_positions  # shape: [num_envs, 2, 3]
        foot_vel = self.foot_velocities  # shape: [num_envs, 2, 3]
        foot_force = self.contact_forces[:, self.feet_indices]  # shape: [num_envs, 2, 3]
        
        # 合并输出
        return torch.cat([
            leg_pos, leg_vel, leg_torque,
            foot_pos.view(self.num_envs, -1),
            foot_vel.view(self.num_envs, -1),
            foot_force.view(self.num_envs, -1)
        ], dim=-1)
    
    def get_arm_obs(self):
        """获取机械臂观测"""
        # 机械臂关节位置、速度、力矩
        arm_pos = self.dof_pos[:, self.num_leg_dofs:]
        arm_vel = self.dof_vel[:, self.num_leg_dofs:]
        arm_torque = self.torques[:, self.num_leg_dofs:] if hasattr(self, "torques") else torch.zeros_like(arm_pos)
        
        # 末端执行器状态
        ee_state = self.end_effector_state  # shape: [num_envs, 7] (x, y, z, qw, qx, qy, qz)
        
        # 机械臂目标状态
        arm_target_pos = self.arm_target_dof_pos
        arm_target_vel = self.arm_target_dof_vel
        
        # 合并输出
        return torch.cat([
            arm_pos, arm_vel, arm_torque, 
            ee_state,
            arm_target_pos, arm_target_vel
        ], dim=-1)
    
    def compute_arm_observations(self):
        """计算机械臂观测"""
        # 更新机械臂关节状态
        self.arm_dof_pos = self.dof_pos[:, self.num_leg_dofs:]
        self.arm_dof_vel = self.dof_vel[:, self.num_leg_dofs:]
        self.arm_torques = self.torques[:, self.num_leg_dofs:] if hasattr(self, "torques") else torch.zeros_like(self.arm_dof_pos)
        
        # 更新末端执行器状态
        self.arm_ee_pos = self.end_effector_state[:, :3]
        self.arm_ee_orn = self.end_effector_state[:, 3:7]
        
        # 计算末端执行器速度（简化计算）
        if hasattr(self, 'last_arm_ee_pos'):
            self.arm_ee_vel[:, :3] = (self.arm_ee_pos - self.last_arm_ee_pos) / self.dt
        else:
            self.arm_ee_vel[:, :3] = torch.zeros_like(self.arm_ee_pos)
        
        self.last_arm_ee_pos = self.arm_ee_pos.clone()
        
        return self.get_arm_obs()
    
    def compute_self_observations(self):
        """重写观测计算，包含机械臂观测"""
        # 基础观测（来自父类）
        obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel, # base_ang_vel 是base的角速度, 3维
            self.projected_gravity, # 映射的重力, 3维
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # dof_pos是关节位置, 14维
            self.dof_vel * self.obs_scales.dof_vel, # dof_vel是关节速度, 14维
            self.actions, # 动作, 14维
            self.clock_inputs_sin.view(self.num_envs, 1), # 时钟, 1维
            self.clock_inputs_cos.view(self.num_envs, 1), # 时钟, 1维
            self.gaits, # 步态, 4维
        ), dim=-1)

        # 机械臂观测
        arm_obs = self.compute_arm_observations()
        
        # 合并基础观测和机械臂观测
        obs_buf = torch.cat([obs_buf, arm_obs], dim=-1)

        # 特权观测（简化版本）
        privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale), dim=1)
        
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(self.cfg.normalization.ground_friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.ground_friction_coeffs.unsqueeze(1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale), dim=1)
       
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale), dim=1)
            
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale), dim=1)
        
        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.com_displacements - com_displacements_shift) * com_displacements_scale), dim=1)
        
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
        
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1)
        
        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, ((self.root_states[:self.num_envs, 2]).view(self.num_envs, -1) - body_height_shift) * body_height_scale), dim=1)

        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.gravities - gravity_shift) / gravity_scale), dim=1)
        
        if self.cfg.env.priv_observe_vel:
            if self.cfg.commands.global_reference:
                privileged_obs_buf = torch.cat((privileged_obs_buf, self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel), dim=-1)
            else:
                privileged_obs_buf = torch.cat((privileged_obs_buf, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
   
        if self.cfg.env.priv_observe_high_freq_goal:
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            self.obj_pose_in_ee.clone(),
                                            self.obj_abg_in_ee.clone()),
                                            dim=1)

        # ee pose and quat dim=7
        lpy = self.get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device))
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        quat_base = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        quat_ee_in_base = quat_mul(quat_base, self.end_effector_state[:, 3:7])
        privileged_obs_buf = torch.cat((privileged_obs_buf, lpy, quat_ee_in_base), dim=1)

        self.privileged_obs_buf = privileged_obs_buf
        
        # 计算critic观测
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, 
            obs_buf
        ), dim=-1)
        
        return obs_buf, critic_obs_buf
    
    def _compute_arm_torques(self, arm_actions):
        """计算机械臂力矩"""
        # 机械臂动作缩放
        arm_actions_scaled = arm_actions * self.cfg.control.action_scale
        
        # 根据控制类型计算力矩
        control_type = self.cfg.control.control_type
        if control_type == "P":
            # 位置控制
            arm_torques = (
                self.arm_p_gains * (arm_actions_scaled + self.arm_default_pos - self.arm_dof_pos)
                - self.arm_d_gains * self.arm_dof_vel
            )
        elif control_type == "V":
            # 速度控制
            arm_torques = (
                self.arm_p_gains * (arm_actions_scaled - self.arm_dof_vel)
                - self.arm_d_gains * (self.arm_dof_vel - self.last_arm_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            # 力矩控制
            arm_torques = arm_actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        return torch.clip(
            arm_torques * self.torques_scale[self.num_leg_dofs:], 
            -self.torque_limits[self.num_leg_dofs:], 
            self.torque_limits[self.num_leg_dofs:]
        )
    
    # def _compute_torques(self, actions):
    #     """重写力矩计算，分别处理腿部和机械臂"""
    #     # 分离腿部和机械臂动作
    #     leg_actions = actions[:, :self.num_leg_dofs]
    #     arm_actions = actions[:, self.num_leg_dofs:]
        
    #     # 计算腿部力矩（直接实现，不调用父类方法）
    #     leg_actions_scaled = leg_actions * self.cfg.control.action_scale
        
    #     # 获取腿部关节的增益
    #     leg_p_gains = self.p_gains[:, :self.num_leg_dofs]
    #     leg_d_gains = self.d_gains[:, :self.num_leg_dofs]
    #     leg_default_pos = self.default_dof_pos[:, :self.num_leg_dofs]
    #     leg_dof_pos = self.dof_pos[:, :self.num_leg_dofs]
    #     leg_dof_vel = self.dof_vel[:, :self.num_leg_dofs]
        
    #     # 腿部PD控制器
    #     leg_torques = (
    #         leg_p_gains * 
    #         (leg_actions_scaled + leg_default_pos - leg_dof_pos)
    #         - leg_d_gains * leg_dof_vel
    #     )
        
    #     # 计算机械臂力矩
    #     arm_torques = self._compute_arm_torques(arm_actions)
        
    #     # 合并力矩
    #     torques = torch.cat([leg_torques, arm_torques], dim=-1)
        
    #     return torch.clip(
    #         torques * self.torques_scale, 
    #         -self.torque_limits, 
    #         self.torque_limits
    #     )
    
    def step(self, actions):
        """重写步进函数，支持机械臂控制"""
        # 动作裁剪
        self._action_clip(actions)
        
        # 步进物理和渲染
        self.render()
        self.pre_physics_step()
        
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            
            # 计算力矩
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            
            # 设置力矩
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

        # 返回裁剪后的观测、奖励、终止标志等
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :5] * self.commands_scale,
            self.critic_obs_buf
        )
    
    def _reward_arm_tracking(self):
        """机械臂跟踪奖励"""
        # 计算末端执行器位置误差
        ee_pos_error = torch.norm(self.arm_ee_pos - self.arm_target_pos, dim=-1)
        return torch.exp(-ee_pos_error / 0.1)
    
    def _reward_arm_energy(self):
        """机械臂能量奖励"""
        # 惩罚机械臂关节速度过大
        arm_vel_penalty = torch.sum(torch.square(self.arm_dof_vel), dim=1)
        return -arm_vel_penalty * 0.01
    
    def _reward_arm_torques(self):
        """机械臂力矩奖励"""
        # 惩罚机械臂力矩过大
        arm_torque_penalty = torch.sum(torch.square(self.arm_torques), dim=1)
        return -arm_torque_penalty * 0.001
    
    def _reward_arm_stability(self):
        """机械臂稳定性奖励"""
        # 惩罚机械臂关节加速度过大
        if hasattr(self, 'last_arm_dof_vel'):
            arm_acc = (self.arm_dof_vel - self.last_arm_dof_vel) / self.dt
            arm_acc_penalty = torch.sum(torch.square(arm_acc), dim=1)
        else:
            arm_acc_penalty = torch.zeros(self.num_envs, device=self.device)
        
        self.last_arm_dof_vel = self.arm_dof_vel.clone()
        return -arm_acc_penalty * 0.01
    
    def compute_reward(self):
        """重写奖励计算，包含机械臂奖励"""
        # 调用父类的奖励计算
        super().compute_reward()
        
        # 添加机械臂相关奖励
        if hasattr(self, 'reward_scales'):
            if 'arm_tracking' in self.reward_scales and self.reward_scales['arm_tracking'] > 0:
                self.rew_buf += self.reward_scales['arm_tracking'] * self._reward_arm_tracking()
            
            if 'arm_energy' in self.reward_scales and self.reward_scales['arm_energy'] > 0:
                self.rew_buf += self.reward_scales['arm_energy'] * self._reward_arm_energy()
            
            if 'arm_torques' in self.reward_scales and self.reward_scales['arm_torques'] > 0:
                self.rew_buf += self.reward_scales['arm_torques'] * self._reward_arm_torques()
            
            if 'arm_stability' in self.reward_scales and self.reward_scales['arm_stability'] > 0:
                self.rew_buf += self.reward_scales['arm_stability'] * self._reward_arm_stability()
    
    def reset_idx(self, env_ids):
        """重写重置函数，包含机械臂重置"""
        # 调用父类重置
        super().reset_idx(env_ids)
        
        # 重置机械臂相关状态
        if len(env_ids) > 0:
            # 重置机械臂关节状态
            self.arm_dof_pos[env_ids] = self.arm_default_pos
            self.arm_dof_vel[env_ids] = 0.0
            self.arm_torques[env_ids] = 0.0
            
            # 重置机械臂目标状态
            self.arm_target_dof_pos[env_ids] = self.arm_default_pos
            self.arm_target_dof_vel[env_ids] = 0.0
            
            # 重置末端执行器状态
            self.arm_ee_pos[env_ids] = 0.0
            self.arm_ee_orn[env_ids] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            self.arm_ee_vel[env_ids] = 0.0
    
    def get_arm_target(self, env_ids):
        """获取机械臂目标状态"""
        return self.arm_target_dof_pos[env_ids], self.arm_target_dof_vel[env_ids]
    
    def set_arm_target(self, env_ids, target_pos, target_vel=None):
        """设置机械臂目标状态"""
        self.arm_target_dof_pos[env_ids] = target_pos
        if target_vel is not None:
            self.arm_target_dof_vel[env_ids] = target_vel
        else:
            self.arm_target_dof_vel[env_ids] = 0.0
    
    def get_arm_state(self, env_ids):
        """获取机械臂当前状态"""
        return self.arm_dof_pos[env_ids], self.arm_dof_vel[env_ids]
    
    def get_ee_state(self, env_ids):
        """获取末端执行器状态"""
        return self.arm_ee_pos[env_ids], self.arm_ee_orn[env_ids], self.arm_ee_vel[env_ids]
    
    def compute_arm_jacobian(self, env_ids):
        """计算机械臂雅可比矩阵（简化版本）"""
        # 这里可以添加实际的雅可比矩阵计算
        # 目前返回单位矩阵作为占位符
        jacobian = torch.eye(6, self.num_arm_dofs, device=self.device).unsqueeze(0).repeat(len(env_ids), 1, 1)
        return jacobian
    
    def inverse_kinematics(self, target_pos, target_orn=None, env_ids=None):
        """逆运动学求解（简化版本）"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 简化的逆运动学，实际应用中需要更复杂的算法
        # 这里只是返回当前关节位置作为占位符
        current_pos = self.arm_dof_pos[env_ids]
        
        # 可以在这里添加实际的逆运动学算法
        # 例如：使用雅可比矩阵的伪逆等
        
        return current_pos
    
    def forward_kinematics(self, joint_positions, env_ids=None):
        """正运动学计算（简化版本）"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 简化的正运动学，实际应用中需要更复杂的算法
        # 这里只是返回末端执行器的当前位置作为占位符
        ee_pos = self.arm_ee_pos[env_ids]
        ee_orn = self.arm_ee_orn[env_ids]
        
        return ee_pos, ee_orn
    
    def get_lpy_in_base_coord(self, env_ids):
        """获取末端执行器在基座坐标系中的位置（l, p, y）"""
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        self.grasper_move = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        self.grasper_move_in_world = quat_rotate(self.end_effector_state[env_ids, 3:7], self.grasper_move)
        self.grasper_in_world = self.end_effector_state[env_ids, :3] + self.grasper_move_in_world

        x = torch.cos(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) \
            + torch.sin(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        y = -torch.sin(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) \
            + torch.cos(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        z = torch.mean(self.grasper_in_world[:, 2].unsqueeze(1) - self.measured_heights, dim=1) - 0.38

        l = torch.sqrt(x**2 + y**2 + z**2)
        p = torch.atan2(z, torch.sqrt(x**2 + y**2))
        y_aw = torch.atan2(y, x)

        return torch.stack([l, p, y_aw], dim=-1)
    
    def _get_ground_frictions(self, env_ids):
        """获取地面摩擦系数"""
        return torch.ones(len(env_ids), device=self.device) * 1.0 