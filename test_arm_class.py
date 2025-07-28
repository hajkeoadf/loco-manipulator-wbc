#!/usr/bin/env python3

import numpy as np
from isaacgym import gymapi, gymtorch
import torch
import sys
import os

# 添加路径
sys.path.append('/home/ril/loco-manipulator-wbc')

from legged_gym.envs.solefoot_flat.solefoot_flat_arm import BipedSFArm
from legged_gym.envs.solefoot_flat.solefoot_flat_arm_config import BipedCfgSFArm

def test_arm_class():
    """测试机械臂子类的功能"""
    print("开始测试机械臂子类...")
    
    # 创建配置
    cfg = BipedCfgSFArm()
    
    # 模拟参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.0025
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # 物理引擎
    physics_engine = gymapi.SIM_PHYSX
    sim_device = "cuda"
    headless = True
    
    try:
        # 创建环境
        print("创建机械臂环境...")
        env = BipedSFArm(cfg, sim_params, physics_engine, sim_device, headless)
        print("✓ 环境创建成功")
        
        # 测试基本属性
        print(f"✓ 腿部关节数量: {env.num_leg_dofs}")
        print(f"✓ 机械臂关节数量: {env.num_arm_dofs}")
        print(f"✓ 总关节数量: {env.num_dofs}")
        print(f"✓ 环境数量: {env.num_envs}")
        
        # 测试观测方法
        print("测试观测方法...")
        leg_obs = env.get_leg_obs()
        arm_obs = env.get_arm_obs()
        print(f"✓ 腿部观测维度: {leg_obs.shape}")
        print(f"✓ 机械臂观测维度: {arm_obs.shape}")
        
        # 测试动作
        print("测试动作执行...")
        actions = torch.randn(env.num_envs, env.num_dofs, device=env.device) * 0.1
        obs, rew, reset, extras, obs_history, commands, critic_obs = env.step(actions)
        print(f"✓ 动作执行成功，观测维度: {obs.shape}")
        print(f"✓ 奖励维度: {rew.shape}")
        
        # 测试机械臂状态获取
        print("测试机械臂状态获取...")
        env_ids = torch.arange(min(10, env.num_envs), device=env.device)
        arm_target_pos, arm_target_vel = env.get_arm_target(env_ids)
        arm_pos, arm_vel = env.get_arm_state(env_ids)
        ee_pos, ee_orn, ee_vel = env.get_ee_state(env_ids)
        print(f"✓ 机械臂目标位置维度: {arm_target_pos.shape}")
        print(f"✓ 机械臂当前位置维度: {arm_pos.shape}")
        print(f"✓ 末端执行器位置维度: {ee_pos.shape}")
        
        # 测试机械臂目标设置
        print("测试机械臂目标设置...")
        new_target = torch.randn(len(env_ids), env.num_arm_dofs, device=env.device) * 0.1
        env.set_arm_target(env_ids, new_target)
        print("✓ 机械臂目标设置成功")
        
        # 测试雅可比矩阵计算
        print("测试雅可比矩阵计算...")
        jacobian = env.compute_arm_jacobian(env_ids)
        print(f"✓ 雅可比矩阵维度: {jacobian.shape}")
        
        # 测试逆运动学
        print("测试逆运动学...")
        target_pos = torch.randn(len(env_ids), 3, device=env.device)
        ik_result = env.inverse_kinematics(target_pos, env_ids=env_ids)
        print(f"✓ 逆运动学结果维度: {ik_result.shape}")
        
        # 测试正运动学
        print("测试正运动学...")
        joint_pos = torch.randn(len(env_ids), env.num_arm_dofs, device=env.device) * 0.1
        fk_pos, fk_orn = env.forward_kinematics(joint_pos, env_ids)
        print(f"✓ 正运动学位置维度: {fk_pos.shape}")
        print(f"✓ 正运动学姿态维度: {fk_orn.shape}")
        
        # 测试奖励计算
        print("测试奖励计算...")
        env.compute_reward()
        print(f"✓ 奖励计算成功，奖励维度: {env.rew_buf.shape}")
        
        # 测试重置
        print("测试环境重置...")
        env.reset_idx(env_ids)
        print("✓ 环境重置成功")
        
        print("\n🎉 所有测试通过！机械臂子类功能正常。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_config():
    """测试配置文件"""
    print("测试配置文件...")
    
    cfg = BipedCfgSFArm()
    
    # 检查关键配置
    assert cfg.env.num_actions == 14, f"动作数量错误: {cfg.env.num_actions}"
    assert cfg.env.num_observations > 54, f"观测维度错误: {cfg.env.num_observations}"
    assert cfg.env.ee_idx == 10, f"末端执行器索引错误: {cfg.env.ee_idx}"
    
    print("✓ 配置文件检查通过")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("机械臂子类测试")
    print("=" * 50)
    
    # 测试配置文件
    if not test_config():
        exit(1)
    
    # 测试机械臂类
    if not test_arm_class():
        exit(1)
    
    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50) 