#!/usr/bin/env python3

import torch
import numpy as np
import os
import sys

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm import BipedSFWithArm
from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm_config import BipedCfgSFWithArm
from legged_gym.algorithm.rsl_actor_critic import ActorCritic
from legged_gym.algorithm.rsl_ppo import PPO
from legged_gym.algorithm.rsl_rollout_storage import RolloutStorage
from legged_gym.algorithm.rsl_on_policy_runner import OnPolicyRunner

def main():
    # 创建配置
    cfg = BipedCfgSFWithArm()
    
    # 创建环境
    env = BipedSFWithArm(
        cfg=cfg,
        sim_params=None,  # 将在环境内部设置
        physics_engine=None,  # 将在环境内部设置
        sim_device='cuda:0',
        headless=False
    )
    
    # 创建Actor-Critic网络
    actor_critic = ActorCritic(
        num_actor_obs=env.get_num_actor_obs(),
        num_critic_obs=env.get_num_critic_obs(),
        num_actions=env.get_num_actions_total(),
        actor_hidden_dims=env.get_actor_hidden_dims(),
        critic_hidden_dims=env.get_critic_hidden_dims(),
        priv_encoder_dims=env.get_priv_encoder_dims(),
        activation=env.get_activation(),
        init_std=env.get_init_std(),
        num_hist=env.get_num_hist(),
        num_prop=env.get_num_prop(),
        num_leg_actions=env.get_num_leg_actions(),
        num_arm_actions=env.get_num_arm_actions(),
        adaptive_arm_gains=env.get_adaptive_arm_gains(),
        adaptive_arm_gains_scale=env.get_adaptive_arm_gains_scale(),
        leg_control_head_hidden_dims=env.get_leg_control_head_hidden_dims(),
        arm_control_head_hidden_dims=env.get_arm_control_head_hidden_dims()
    )
    
    # 创建PPO算法
    ppo = PPO(
        actor_critic=actor_critic,
        value_loss_coef=cfg.algorithm.value_loss_coef,
        use_clipped_value_loss=cfg.algorithm.use_clipped_value_loss,
        clip_param=cfg.algorithm.clip_param,
        entropy_coef=cfg.algorithm.entropy_coef,
        num_learning_epochs=cfg.algorithm.num_learning_epochs,
        num_mini_batches=cfg.algorithm.num_mini_batches,
        learning_rate=cfg.algorithm.learning_rate,
        schedule=cfg.algorithm.schedule,
        gamma=cfg.algorithm.gamma,
        lam=cfg.algorithm.lam,
        desired_kl=cfg.algorithm.desired_kl,
        max_grad_norm=cfg.algorithm.max_grad_norm
    )
    
    # 创建存储
    storage = RolloutStorage(
        num_envs=cfg.env.num_envs,
        num_transitions_per_env=cfg.runner.num_steps_per_env,
        actor_obs_shape=(env.get_num_actor_obs(),),
        critic_obs_shape=(env.get_num_critic_obs(),),
        actions_shape=(env.get_num_actions_total(),),
        device='cuda:0'
    )
    
    # 创建训练器
    runner = OnPolicyRunner(
        env=env,
        actor_critic=actor_critic,
        ppo=ppo,
        storage=storage,
        device='cuda:0',
        num_envs=cfg.env.num_envs,
        num_transitions_per_env=cfg.runner.num_steps_per_env,
        max_iterations=cfg.runner.max_iterations,
        save_interval=cfg.runner.save_interval,
        experiment_name=cfg.runner.experiment_name,
        run_name=cfg.runner.run_name,
        load_run=cfg.runner.load_run,
        checkpoint=cfg.runner.checkpoint,
        resume=cfg.runner.resume
    )
    
    # 开始训练
    runner.run()

if __name__ == "__main__":
    main() 