#!/usr/bin/env python3
"""
机械臂运动验证脚本
用于确认Isaac Gym中的机械臂能够正常运动
"""

import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import isaacgym

def test_arm_motion():
    print("\n🚀 启动机械臂运动验证程序")
    print("="*60)
    
    # 设置参数
    args = get_args()
    args.task = "solefoot_flat_with_arm"
    args.headless = False  # 显示可视化
    args.num_envs = 4  # 减少环境数量以便观察
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    print("✅ 环境创建成功")
    
    # 运行验证测试
    try:
        print("\n📋 开始基础验证...")
        
        # 1. 重置环境
        obs = env.reset()
        print("✅ 环境重置成功")
        
        # 2. 验证动作空间
        print(f"✅ 观察空间维度: {obs.shape}")
        print(f"✅ 动作空间维度: {env.num_actions}")
        print(f"✅ 机械臂动作数: {env.get_num_arm_actions()}")
        print(f"✅ 腿部动作数: {env.get_num_leg_actions()}")
        
        # 3. 运行一些随机动作来观察运动
        print("\n🎮 执行随机动作测试...")
        for step in range(200):
            # 生成随机动作
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            
            # 腿部动作：小幅随机
            actions[:, :8] = 0.1 * (torch.rand_like(actions[:, :8]) - 0.5)
            
            # 机械臂动作：较大幅度的周期性运动
            for i in range(6):
                actions[:, 8+i] = 0.5 * torch.sin(2 * np.pi * step / 50 + i * np.pi/3)
            
            # 执行动作
            obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
            
            # 每50步输出一次状态
            if step % 50 == 0:
                print(f"步数 {step}: 腿部奖励={rew[0].item():.3f}, 机械臂奖励={arm_rew[0].item():.3f}")
        
        print("\n🧪 运行专门的机械臂运动测试...")
        # 4. 运行专门的机械臂测试
        env.test_arm_motion(test_duration=300)
        
        print("\n🎯 测试任务跟踪...")
        # 5. 测试目标跟踪
        for step in range(100):
            # 使用零动作，让系统尝试达到目标
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
        
        print("✅ 所有测试完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔚 测试结束，查看上面的输出以确认机械臂运动状态")

if __name__ == "__main__":
    test_arm_motion()
