#!/usr/bin/env python3
import numpy as np
import os
import sys
from isaacgym import gymapi

# 添加路径 - 需要添加项目根目录到Python路径
# current_dir = os.path.dirname(__file__)
# project_root = os.path.join(current_dir, '..', '..')  # 回到项目根目录
# sys.path.append(project_root)
# sys.path.append(os.path.join(project_root, 'legged_gym'))

# 先导入IsaacGym相关模块
from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm import BipedSFWithArm
from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm_config import BipedCfgSFWithArm

# 在IsaacGym模块导入后再导入torch
import torch


def test_environment():
    """测试环境是否正常工作"""
    print("开始测试solefoot_flat_with_arm环境...")
    
    # 创建配置
    cfg = BipedCfgSFWithArm()
    
    # 减少环境数量用于测试
    cfg.env.num_envs = 4

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
    
    try:
        # 创建环境
        print("创建环境...")
        env = BipedSFWithArm(cfg, sim_params, physics_engine, sim_device, headless)
        print("环境创建成功!")
        
        # 测试观察空间
        print(f"观察空间维度: {env.get_num_observations()}")
        print(f"动作空间维度: {env.get_num_actions()}")
        print(f"腿部动作维度: {env.get_num_leg_actions()}")
        print(f"机械臂动作维度: {env.get_num_arm_actions()}")
        
        # 测试重置
        print("测试环境重置...")
        obs = env.reset()
        print(f"重置后观察形状: {obs.shape}")
        
        # 测试步进
        print("测试环境步进...")
        for i in range(10):
            # 生成随机动作
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
            
            # 执行步进
            obs, privileged_obs, rewards, dones, _, _, _ = env.step(actions)
            
            print(f"步骤 {i+1}:")
            print(f"  观察形状: {obs.shape}")
            print(f"  奖励形状: {rewards.shape}")
            print(f"  完成标志: {dones}")
            print(f"  奖励: {rewards}")
            # print(f"  完成标志形状: {dones.shape}")
            # print(f"  平均奖励: {rewards.mean().item():.4f}")
            
            # # 检查是否有环境需要重置
            # if dones.any():
            #     print(f"  有 {dones.sum().item()} 个环境需要重置")
        
        print("环境测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_rsl_compatibility():
    """测试与RSL算法的兼容性"""
    print("\n开始测试RSL算法兼容性...")
    
    # 创建配置
    cfg = BipedCfgSFWithArm()
    cfg.env.num_envs = 4

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
    
    try:
        # 创建环境
        env = BipedSFWithArm(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 测试RSL算法所需的接口
        print("测试RSL算法接口...")
        
        # 获取各种维度信息
        num_actor_obs = env.get_num_actor_obs()
        num_critic_obs = env.get_num_critic_obs()
        num_actions = env.get_num_actions_total()
        num_leg_actions = env.get_num_leg_actions()
        num_arm_actions = env.get_num_arm_actions()
        
        print(f"Actor观察维度: {num_actor_obs}")
        print(f"Critic观察维度: {num_critic_obs}")
        print(f"总动作维度: {num_actions}")
        print(f"腿部动作维度: {num_leg_actions}")
        print(f"机械臂动作维度: {num_arm_actions}")
        
        # 验证维度一致性
        assert num_actions == num_leg_actions + num_arm_actions, "动作维度不匹配"
        print("动作维度验证通过!")
        
        # 测试网络参数
        actor_hidden_dims = env.get_actor_hidden_dims()
        critic_hidden_dims = env.get_critic_hidden_dims()
        priv_encoder_dims = env.get_priv_encoder_dims()
        
        print(f"Actor隐藏层维度: {actor_hidden_dims}")
        print(f"Critic隐藏层维度: {critic_hidden_dims}")
        print(f"特权编码器维度: {priv_encoder_dims}")
        
        # 测试观察获取
        obs = env.get_observations()
        privileged_obs = env.get_privileged_observations()
        rewards = env.get_rewards()
        arm_rewards = env.get_arm_rewards()
        dones = env.get_dones()
        extras = env.get_extras()
        
        print(f"观察形状: {obs.shape}")
        print(f"特权观察形状: {privileged_obs.shape}")
        print(f"奖励形状: {rewards.shape}")
        print(f"机械臂奖励形状: {arm_rewards.shape}")
        print(f"完成标志形状: {dones.shape}")
        
        print("RSL算法兼容性测试完成!")
        
    except Exception as e:
        print(f"RSL兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment()
    # test_rsl_compatibility() 