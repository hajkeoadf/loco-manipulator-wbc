#!/usr/bin/env python3
import isaacgym
from isaacgym import gymapi
import torch
import numpy as np
import os
import sys

from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm import BipedSFWithArm
from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm_config import BipedCfgSFWithArm, BipedCfgPPOSFWithArm
from legged_gym.algorithm.rsl_actor_critic import ActorCritic
from legged_gym.algorithm.rsl_ppo import RSLPPO
from legged_gym.algorithm.rsl_rollout_storage import RolloutStorage
from legged_gym.algorithm.rsl_on_policy_runner import RSLOnPolicyRunner
from legged_gym.utils.helpers import class_to_dict

def main():
    # 创建配置
    cfg = BipedCfgSFWithArm()
    ppo_cfg = BipedCfgPPOSFWithArm()

    # 模拟参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.0025
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # 物理引擎
    physics_engine = gymapi.SIM_PHYSX
    sim_device = "cuda"
    headless = False
    
    # 创建环境
    env = BipedSFWithArm(
        cfg=cfg,
        sim_params=sim_params,  # 将在环境内部设置
        physics_engine=physics_engine,  # 将在环境内部设置
        sim_device=sim_device,
        headless=headless
    )
    
    # # 创建Actor-Critic网络
    # actor_critic = ActorCritic(
    #     num_actor_obs=env.get_num_actor_obs(),
    #     num_critic_obs=env.get_num_critic_obs(),
    #     num_actions=env.get_num_actions_total(),
    #     actor_hidden_dims=env.get_actor_hidden_dims(),
    #     critic_hidden_dims=env.get_critic_hidden_dims(),
    #     priv_encoder_dims=env.get_priv_encoder_dims(),
    #     activation=env.get_activation(),
    #     init_std=env.get_init_std(),
    #     num_hist=env.get_num_hist(),
    #     num_prop=env.get_num_prop(),
    #     num_leg_actions=env.get_num_leg_actions(),
    #     num_arm_actions=env.get_num_arm_actions(),
    #     adaptive_arm_gains=env.get_adaptive_arm_gains(),
    #     adaptive_arm_gains_scale=env.get_adaptive_arm_gains_scale(),
    #     leg_control_head_hidden_dims=env.get_leg_control_head_hidden_dims(),
    #     arm_control_head_hidden_dims=env.get_arm_control_head_hidden_dims()
    # )
    
    # # 创建PPO算法
    # ppo = RSLPPO(
    #     actor_critic=actor_critic,
    #     value_loss_coef=ppo_cfg.algorithm.value_loss_coef,
    #     use_clipped_value_loss=ppo_cfg.algorithm.use_clipped_value_loss,
    #     clip_param=ppo_cfg.algorithm.clip_param,
    #     entropy_coef=ppo_cfg.algorithm.entropy_coef,
    #     num_learning_epochs=ppo_cfg.algorithm.num_learning_epochs,
    #     num_mini_batches=ppo_cfg.algorithm.num_mini_batches,
    #     learning_rate=ppo_cfg.algorithm.learning_rate,
    #     schedule=ppo_cfg.algorithm.schedule,
    #     gamma=ppo_cfg.algorithm.gamma,
    #     lam=ppo_cfg.algorithm.lam,
    #     desired_kl=ppo_cfg.algorithm.desired_kl,
    #     max_grad_norm=ppo_cfg.algorithm.max_grad_norm
    # )
    
    # # 创建存储
    # storage = RolloutStorage(
    #     num_envs=cfg.env.num_envs,
    #     num_transitions_per_env=ppo_cfg.runner.num_steps_per_env,
    #     actor_obs_shape=(env.get_num_actor_obs(),),
    #     critic_obs_shape=(env.get_num_critic_obs(),),
    #     actions_shape=(env.get_num_actions_total(),),
    #     device='cuda:0'
    # )

    train_cfg = class_to_dict(ppo_cfg)

    # 创建训练器
    runner = RSLOnPolicyRunner(
        env=env, 
        train_cfg=train_cfg, 
        log_dir=None, 
        device=sim_device)
    
    # 开始训练
    runner.learn(num_learning_iterations=ppo_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main() 