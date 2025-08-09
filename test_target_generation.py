 #!/usr/bin/env python3
"""
目标生成测试
验证机械臂目标生成的改进，确保目标位置更合理
"""
import isaacgym
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def test_target_generation():
    print("\n🎯 机械臂目标生成测试")
    print("="*60)
    print("验证黄色球（目标）和蓝色球（当前位置）的相对位置")
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
    
    print("🔧 测试目标生成范围:")
    print(f"  初始球坐标范围:")
    print(f"    长度 (r): {env_cfg.goal_ee.ranges.init_pos_l}")
    print(f"    俯仰角 (θ): {env_cfg.goal_ee.ranges.init_pos_p} rad")
    print(f"    方位角 (φ): {env_cfg.goal_ee.ranges.init_pos_y} rad")
    
    theta_deg = [np.degrees(env_cfg.goal_ee.ranges.init_pos_p[0]), np.degrees(env_cfg.goal_ee.ranges.init_pos_p[1])]
    print(f"    俯仰角 (θ): {theta_deg} 度")
    
    print(f"\n  碰撞检测范围:")
    print(f"    下限: {env_cfg.goal_ee.collision_lower_limits}")
    print(f"    上限: {env_cfg.goal_ee.collision_upper_limits}")
    print(f"    地面限制: {env_cfg.goal_ee.underground_limit}")
    
    print(f"\n  初始起始位置:")
    init_cart = torch.tensor([0.3, 0.0, 0.25])  # 我们设置的新起始位置
    print(f"    笛卡尔坐标: [{init_cart[0]:.3f}, {init_cart[1]:.3f}, {init_cart[2]:.3f}]")
    
    # 转换为球坐标看看
    r = torch.sqrt(torch.sum(init_cart**2))
    theta = torch.acos(init_cart[2] / r)
    phi = torch.atan2(init_cart[1], init_cart[0])
    print(f"    球坐标: r={r:.3f}, θ={theta:.3f} rad ({np.degrees(theta):.1f}°), φ={phi:.3f} rad ({np.degrees(phi):.1f}°)")
    
    print("\n🎮 进行目标生成测试...")
    
    target_positions = []
    ee_positions = []
    height_differences = []
    
    # 进行多次目标重采样，收集统计数据
    for i in range(10):
        print(f"\n--- 第 {i+1} 次重采样 ---")
        
        # 重采样目标
        env._resample_ee_goal(torch.tensor([0], device=env.device), is_init=True)
        
        # 执行几步让系统稳定
        for _ in range(5):
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
        
        # 获取当前状态
        ee_pos = env.ee_pos[0].cpu().numpy()
        target_pos = env.curr_ee_goal_cart[0].cpu().numpy()
        target_sphere = env.curr_ee_goal_sphere[0].cpu().numpy()
        
        # 计算在世界坐标系中的目标位置
        base_pos = env.root_states[0, :3].cpu().numpy()
        target_world = base_pos + target_pos
        
        height_diff = target_world[2] - ee_pos[2]
        
        print(f"  末端执行器位置: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]")
        print(f"  目标本地坐标:   [{target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:+.3f}]")
        print(f"  目标世界坐标:   [{target_world[0]:+.3f}, {target_world[1]:+.3f}, {target_world[2]:+.3f}]")
        print(f"  目标球坐标:     [r={target_sphere[0]:.3f}, θ={target_sphere[1]:.3f} ({np.degrees(target_sphere[1]):.1f}°), φ={target_sphere[2]:.3f} ({np.degrees(target_sphere[2]):.1f}°)]")
        print(f"  高度差 (目标-当前): {height_diff:+.3f} m {'✅ 目标更高' if height_diff > 0 else '❌ 目标更低' if height_diff < -0.05 else '⚖️  高度相近'}")
        
        target_positions.append(target_world.copy())
        ee_positions.append(ee_pos.copy())
        height_differences.append(height_diff)
        
        # 等待观察
        if i < 3:  # 前3次等待更长时间以便观察
            print("  (等待3秒以便观察可视化...)")
            for _ in range(1200):  # 3秒 @ 400Hz
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
    
    # 统计分析
    height_differences = np.array(height_differences)
    target_positions = np.array(target_positions)
    ee_positions = np.array(ee_positions)
    
    print("\n" + "="*60)
    print("📊 统计分析结果")
    print("="*60)
    
    print(f"高度差统计 (目标 - 当前末端执行器):")
    print(f"  平均值: {np.mean(height_differences):+.3f} m")
    print(f"  标准差: {np.std(height_differences):.3f} m")
    print(f"  最小值: {np.min(height_differences):+.3f} m")
    print(f"  最大值: {np.max(height_differences):+.3f} m")
    
    positive_height_ratio = np.sum(height_differences > 0) / len(height_differences)
    print(f"  目标更高的比例: {positive_height_ratio:.1%}")
    
    print(f"\n目标位置统计 (世界坐标系):")
    print(f"  X范围: [{np.min(target_positions[:, 0]):+.3f}, {np.max(target_positions[:, 0]):+.3f}] m")
    print(f"  Y范围: [{np.min(target_positions[:, 1]):+.3f}, {np.max(target_positions[:, 1]):+.3f}] m")
    print(f"  Z范围: [{np.min(target_positions[:, 2]):+.3f}, {np.max(target_positions[:, 2]):+.3f}] m")
    print(f"  Z平均: {np.mean(target_positions[:, 2]):+.3f} m")
    
    print(f"\n末端执行器位置统计:")
    print(f"  X范围: [{np.min(ee_positions[:, 0]):+.3f}, {np.max(ee_positions[:, 0]):+.3f}] m")
    print(f"  Y范围: [{np.min(ee_positions[:, 1]):+.3f}, {np.max(ee_positions[:, 1]):+.3f}] m")
    print(f"  Z范围: [{np.min(ee_positions[:, 2]):+.3f}, {np.max(ee_positions[:, 2]):+.3f}] m")
    print(f"  Z平均: {np.mean(ee_positions[:, 2]):+.3f} m")
    
    print("\n📋 改进评估:")
    if positive_height_ratio > 0.6:
        print("✅ 很好！大多数目标位置高于当前末端执行器位置")
    elif positive_height_ratio > 0.4:
        print("⚖️  可以接受，目标位置高度分布相对均匀")
    else:
        print("❌ 需要进一步调整，太多目标位置低于当前位置")
    
    if np.mean(height_differences) > 0.05:
        print("✅ 很好！目标位置平均高度高于当前位置")
    elif np.mean(height_differences) > -0.05:
        print("⚖️  可以接受，目标位置平均高度接近当前位置")
    else:
        print("❌ 目标位置平均过低，建议进一步调整")
    
    print("\n🎯 建议:")
    if positive_height_ratio < 0.6:
        print("- 可以进一步增加球坐标俯仰角范围，使目标更多出现在水平或稍高位置")
        print("- 考虑调整初始起始位置或目标生成策略")
    
    if np.std(height_differences) > 0.3:
        print("- 目标高度变化较大，可以考虑缩小球坐标俯仰角范围以获得更一致的目标高度")
    
    print("\n✅ 目标生成测试完成！")


if __name__ == "__main__":
    test_target_generation()