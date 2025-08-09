#!/usr/bin/env python3
"""
PD控制演示
展示为什么零动作时仍会有扭矩和速度
"""
import isaacgym
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def demo_pd_control():
    print("\n🎮 PD控制原理演示")
    print("="*60)
    print("展示为什么动作为0时仍会有扭矩和速度")
    print("="*60)
    
    # 设置参数
    args = get_args()
    args.task = "solefoot_flat_with_arm"
    args.headless = False
    args.num_envs = 1
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    obs = env.reset()
    
    print("✅ 环境创建成功\n")
    
    # 演示1：给一个大的初始动作，然后设为零
    print("🔧 演示1: 给机械臂一个扰动，然后观察零动作下的行为")
    print("-" * 60)
    
    # 给J1一个大的扰动
    for step in range(20):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        actions[:, 8] = 1.0  # J1最大正动作
        obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
    
    print("扰动结束，现在应用零动作并观察PD控制器的响应:")
    print()
    
    # 现在应用零动作，观察PD控制器如何工作
    for step in range(50):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)  # 所有动作为0
        obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
        
        if step % 10 == 0:
            # 获取J1状态
            j1_pos = env.dof_pos[0, 8].item()
            j1_vel = env.dof_vel[0, 8].item()
            j1_action = actions[0, 8].item()
            j1_torque = env.torques[0, 8].item()
            
            # 计算PD控制器的理论输出
            default_pos = env.default_dof_pos[0, 8].item()  # 默认位置
            pos_error = default_pos - j1_pos
            kp = env.p_gains[0, 8].item()  # P增益
            kd = env.d_gains[0, 8].item()  # D增益
            
            theoretical_torque = kp * pos_error - kd * j1_vel
            
            print(f"步数 {step:2d}:")
            print(f"  位置: {j1_pos:+.3f} rad (默认: {default_pos:+.3f}, 误差: {pos_error:+.3f})")
            print(f"  速度: {j1_vel:+.3f} rad/s")
            print(f"  动作: {j1_action:+.3f}")
            print(f"  扭矩: {j1_torque:+.3f} Nm (理论: {theoretical_torque:+.3f})")
            print(f"  PD分解: Kp×误差={kp*pos_error:+.3f}, -Kd×速度={-kd*j1_vel:+.3f}")
            print()
    
    print("🔧 演示2: 重力补偿演示")
    print("-" * 60)
    print("观察机械臂在零动作下如何对抗重力:\n")
    
    # 重置到一个受重力影响较大的姿态
    env.reset()
    
    # 将机械臂设置到一个水平伸展的姿态（受重力影响大）
    target_positions = torch.tensor([0.0, 1.57, 0.0, 0.0, 0.0, 0.0], device=env.device)  # J2水平
    for step in range(30):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for i in range(6):
            pos_error = target_positions[i] - env.dof_pos[0, 8+i]
            actions[0, 8+i] = torch.clamp(pos_error * 2.0, -1.0, 1.0)  # 简单比例控制
        obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
    
    print("机械臂已调整到水平姿态，现在观察零动作下的重力补偿:")
    print()
    
    # 现在零动作，观察重力补偿
    for step in range(30):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
        
        if step % 10 == 0:
            print(f"步数 {step:2d} - 零动作下的状态:")
            for i, joint_name in enumerate(["J1", "J2", "J3", "J4", "J5", "J6"]):
                pos = env.dof_pos[0, 8+i].item()
                vel = env.dof_vel[0, 8+i].item()
                torque = env.torques[0, 8+i].item()
                print(f"  {joint_name}: pos={pos:+.3f}, vel={vel:+.3f}, torque={torque:+.3f}")
            print()
    
    print("🔧 演示3: 关节耦合效应")
    print("-" * 60)
    print("观察一个关节运动如何影响其他关节:\n")
    
    env.reset()
    
    # 只让J2运动，观察其他关节的响应
    for step in range(50):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        # 只有J2有动作
        phase = torch.tensor(2 * np.pi * step / 25, device=env.device)
        actions[:, 9] = 0.5 * torch.sin(phase)  # J2动作
        
        obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
        
        if step % 15 == 0:
            print(f"步数 {step:2d} - 只有J2动作时:")
            for i, joint_name in enumerate(["J1", "J2", "J3", "J4", "J5", "J6"]):
                pos = env.dof_pos[0, 8+i].item()
                vel = env.dof_vel[0, 8+i].item()
                action = actions[0, 8+i].item()
                torque = env.torques[0, 8+i].item()
                
                marker = "🎯" if action != 0 else ("⚙️" if abs(torque) > 0.1 else "  ")
                print(f"  {marker} {joint_name}: action={action:+.3f}, vel={vel:+.3f}, torque={torque:+.3f}")
            print()
    
    print("✅ PD控制演示完成!")
    print("\n📋 总结:")
    print("🔹 即使动作为0，PD控制器仍会:")
    print("   • 试图将关节拉回默认位置 (位置控制)")
    print("   • 提供阻尼以减缓运动 (速度阻尼)")
    print("   • 补偿重力和其他外力")
    print("   • 响应其他关节的耦合效应")
    print()
    print("🔹 这是正常的机器人控制行为，不是bug!")
    print("🔹 扭矩 = Kp×(默认位置-当前位置) - Kd×当前速度")


if __name__ == "__main__":
    demo_pd_control() 