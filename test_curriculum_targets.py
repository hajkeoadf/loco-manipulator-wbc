#!/usr/bin/env python3
"""
测试机械臂课程学习目标生成策略
验证目标是否始终在末端执行器前方，以及课程进度是否正确
"""
import isaacgym
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from legged_gym.envs.solefoot_flat_with_arm.solefoot_flat_with_arm import BipedSFWithArm
from legged_gym.envs.solefoot_flat_with_arm.solefoot_flat_with_arm_config import BipedCfgSFWithArm

# 设置英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def visualize_targets_3d(targets_cart, targets_sphere, step, progress, save_path=None):
    """3D可视化目标位置"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D散点图
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(targets_cart[:, 0], targets_cart[:, 1], targets_cart[:, 2], 
                         c=targets_sphere[:, 0], cmap='viridis', s=50, alpha=0.7)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Distance (m)')
    
    # 设置坐标轴
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Left-Right)')
    ax1.set_zlabel('Z (Height)')
    ax1.set_title(f'3D Target Distribution (Curriculum Progress: {progress:.1f}%)')
    
    # 添加机器人位置标记
    ax1.scatter([0], [0], [0], color='red', s=100, marker='o', label='Robot Position')
    ax1.legend()
    
    # 添加坐标轴范围
    ax1.set_xlim([-0.5, 1.0])
    ax1.set_ylim([-0.5, 0.5])
    ax1.set_zlim([0, 1.0])
    
    # 俯视图 (X-Y平面)
    ax2 = fig.add_subplot(222)
    ax2.scatter(targets_cart[:, 0], targets_cart[:, 1], c=targets_sphere[:, 0], 
                cmap='viridis', s=50, alpha=0.7)
    ax2.scatter([0], [0], color='red', s=100, marker='o', label='Robot Position')
    ax2.set_xlabel('X (Forward)')
    ax2.set_ylabel('Y (Left-Right)')
    ax2.set_title('Top View (X-Y Plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 侧视图 (X-Z平面)
    ax3 = fig.add_subplot(223)
    ax3.scatter(targets_cart[:, 0], targets_cart[:, 2], c=targets_sphere[:, 0], 
                cmap='viridis', s=50, alpha=0.7)
    ax3.scatter([0], [0], color='red', s=100, marker='o', label='Robot Position')
    ax3.set_xlabel('X (Forward)')
    ax3.set_ylabel('Z (Height)')
    ax3.set_title('Side View (X-Z Plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 球坐标分布
    ax4 = fig.add_subplot(224)
    # 距离分布直方图
    ax4.hist(targets_sphere[:, 0], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Target Distance Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()

def visualize_curriculum_progress(env, test_steps, save_path=None):
    """可视化课程进度"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 收集数据
    progress_data = []
    distance_ranges = []
    forward_ratios = []
    
    for step in test_steps:
        env.curriculum_step = step
        progress = (step / env.max_curriculum_step) * 100
        progress_data.append(progress)
        
        env_ids = torch.arange(env.num_envs, device=env.device)
        env._resample_ee_goal(env_ids, is_init=False)
        
        targets_cart = env.ee_goal_cart.cpu().numpy()
        targets_sphere = env.ee_goal_sphere.cpu().numpy()
        
        # 计算距离范围
        distance_ranges.append([np.min(targets_sphere[:, 0]), np.max(targets_sphere[:, 0])])
        
        # 计算前方目标比例
        forward_count = np.sum(targets_cart[:, 0] > 0.1)
        forward_ratios.append(forward_count / env.num_envs * 100)
    
    # 课程进度图
    ax1 = axes[0, 0]
    ax1.plot(progress_data, test_steps, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Curriculum Progress (%)')
    ax1.set_ylabel('Training Steps')
    ax1.set_title('Curriculum Progress vs Training Steps')
    ax1.grid(True, alpha=0.3)
    
    # 距离范围变化
    ax2 = axes[0, 1]
    distance_ranges = np.array(distance_ranges)
    ax2.fill_between(progress_data, distance_ranges[:, 0], distance_ranges[:, 1], 
                     alpha=0.3, color='green', label='Actual Distance Range')
    
    # 添加期望距离范围
    expected_l_min = [0.15 + (p/100) * (0.25 - 0.15) for p in progress_data]
    expected_l_max = [0.25 + (p/100) * (0.6 - 0.25) for p in progress_data]
    ax2.fill_between(progress_data, expected_l_min, expected_l_max, 
                     alpha=0.2, color='red', label='Expected Distance Range')
    
    ax2.set_xlabel('Curriculum Progress (%)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Distance Range vs Curriculum Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 前方目标比例
    ax3 = axes[1, 0]
    ax3.bar(progress_data, forward_ratios, alpha=0.7, color='orange')
    ax3.set_xlabel('Curriculum Progress (%)')
    ax3.set_ylabel('Forward Target Ratio (%)')
    ax3.set_title('Forward Target Ratio')
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3)
    
    # 添加100%基准线
    ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal Value')
    ax3.legend()
    
    # 目标分布统计
    ax4 = axes[1, 1]
    # 使用最后一个测试步骤的数据
    env.curriculum_step = test_steps[-1]
    env_ids = torch.arange(env.num_envs, device=env.device)
    env._resample_ee_goal(env_ids, is_init=False)
    targets_cart = env.ee_goal_cart.cpu().numpy()
    
    # 创建2D直方图
    ax4.hist2d(targets_cart[:, 0], targets_cart[:, 1], bins=10, cmap='Blues')
    ax4.scatter([0], [0], color='red', s=100, marker='o', label='Robot Position')
    ax4.set_xlabel('X (Forward)')
    ax4.set_ylabel('Y (Left-Right)')
    ax4.set_title('Target Position Density Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curriculum progress chart saved to: {save_path}")
    
    plt.show()

def test_curriculum_targets():
    """测试课程学习目标生成"""
    print("🧪 测试机械臂课程学习目标生成策略")
    print("="*60)
    
    # 创建环境
    cfg = BipedCfgSFWithArm()
    cfg.env.num_envs = 4  # 增加环境数量以获得更好的可视化效果
    cfg.env.num_observations = 65
    cfg.env.num_privileged_obs = 5
    cfg.env.num_actions = 14
    cfg.control.decimation = 4
    cfg.domain_rand.push_robots = False
    
    # 创建环境实例
    from isaacgym import gymapi, gymutil
    
    # 创建必要的参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.01
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True
    
    physics_engine = gymapi.SIM_PHYSX
    
    env = BipedSFWithArm(cfg, sim_params, physics_engine, "cuda", False)
    
    print(f"✅ 环境创建成功")
    print(f"   环境数量: {env.num_envs}")
    print(f"   动作维度: {env.num_actions}")
    print(f"   观察维度: {env.num_obs}")
    
    # 重置环境
    obs = env.reset()
    print(f"✅ 环境重置成功")
    
    # 测试不同课程阶段的目标生成
    test_steps = [0, 1000, 5000, 10000]
    
    for i, step in enumerate(test_steps):
        print(f"\n🎯 测试课程步数: {step}")
        print("-" * 40)
        
        # 设置课程步数
        env.curriculum_step = step
        progress = (step / env.max_curriculum_step) * 100
        print(f"课程进度: {progress:.1f}%")
        
        # 生成新目标
        env_ids = torch.arange(env.num_envs, device=env.device)
        env._resample_ee_goal(env_ids, is_init=False)
        
        # 分析生成的目标
        targets_cart = env.ee_goal_cart.cpu().numpy()
        targets_sphere = env.ee_goal_sphere.cpu().numpy()
        
        print(f"目标统计:")
        print(f"  X范围: [{np.min(targets_cart[:, 0]):.3f}, {np.max(targets_cart[:, 0]):.3f}] m")
        print(f"  Y范围: [{np.min(targets_cart[:, 1]):.3f}, {np.max(targets_cart[:, 1]):.3f}] m")
        print(f"  Z范围: [{np.min(targets_cart[:, 2]):.3f}, {np.max(targets_cart[:, 2]):.3f}] m")
        
        # 检查是否所有目标都在前方
        forward_targets = np.sum(targets_cart[:, 0] > 0.1)
        total_targets = len(targets_cart)
        print(f"前方目标: {forward_targets}/{total_targets} ({forward_targets/total_targets*100:.1f}%)")
        
        # 显示球坐标信息
        print(f"球坐标统计:")
        print(f"  半径范围: [{np.min(targets_sphere[:, 0]):.3f}, {np.max(targets_sphere[:, 0]):.3f}] m")
        print(f"  俯仰角范围: [{np.degrees(np.min(targets_sphere[:, 1])):.1f}°, {np.degrees(np.max(targets_sphere[:, 1])):.1f}°]")
        print(f"  偏航角范围: [{np.degrees(np.min(targets_sphere[:, 2])):.1f}°, {np.degrees(np.max(targets_sphere[:, 2])):.1f}°]")
        
        # 验证目标质量
        if forward_targets == total_targets:
            print("  ✅ 所有目标都在前方")
        else:
            print("  ❌ 存在后方目标")
            
        # 检查距离范围是否符合课程进度
        expected_l_min = 0.15 + (progress/100) * (0.25 - 0.15)
        expected_l_max = 0.25 + (progress/100) * (0.6 - 0.25)
        actual_l_min = np.min(targets_sphere[:, 0])
        actual_l_max = np.max(targets_sphere[:, 0])
        
        if actual_l_min >= expected_l_min and actual_l_max <= expected_l_max:
            print("  ✅ 距离范围符合课程进度")
        else:
            print(f"  ⚠️  距离范围: 期望[{expected_l_min:.3f}, {expected_l_max:.3f}], 实际[{actual_l_min:.3f}, {actual_l_max:.3f}]")
        
        # 可视化当前阶段的目标
        print(f"\n📊 Generating visualization charts...")
        # visualize_targets_3d(targets_cart, targets_sphere, step, progress, 
        #                    save_path=f"targets_visualization_step_{step}.png")
    
    print("\n" + "="*60)
    print("🎯 课程学习目标生成测试完成")
    print("="*60)
    
    # 生成课程进度综合可视化
    print("\n📈 Generating curriculum progress visualization...")
    visualize_curriculum_progress(env, test_steps, save_path="curriculum_progress_visualization.png")
    
    # 测试连续目标生成的一致性
    print("\n🔄 测试连续目标生成的一致性")
    print("-" * 40)
    
    env.curriculum_step = 5000  # 中等难度
    consistency_tests = 10
    
    all_targets_x = []
    all_targets_forward = []
    
    for i in range(consistency_tests):
        env_ids = torch.arange(env.num_envs, device=env.device)
        env._resample_ee_goal(env_ids, is_init=False)
        
        targets_cart = env.ee_goal_cart.cpu().numpy()
        forward_count = np.sum(targets_cart[:, 0] > 0.1)
        
        all_targets_x.extend(targets_cart[:, 0])
        all_targets_forward.append(forward_count)
        
        print(f"测试 {i+1}: 前方目标 {forward_count}/{env.num_envs}")
    
    # 统计一致性
    forward_ratio = np.mean(all_targets_forward) / env.num_envs * 100
    x_positive_ratio = np.sum(np.array(all_targets_x) > 0.1) / len(all_targets_x) * 100
    
    print(f"\n一致性统计:")
    print(f"  前方目标比例: {forward_ratio:.1f}%")
    print(f"  X坐标正比例: {x_positive_ratio:.1f}%")
    
    if forward_ratio > 95 and x_positive_ratio > 95:
        print("  ✅ 目标生成一致性良好")
    else:
        print("  ⚠️  目标生成一致性需要改进")
    
    # 可视化一致性测试结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 前方目标比例变化
    ax1.plot(range(1, consistency_tests + 1), 
             [f/env.num_envs*100 for f in all_targets_forward], 'bo-')
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal Value')
    ax1.set_xlabel('Test Number')
    ax1.set_ylabel('Forward Target Ratio (%)')
    ax1.set_title('Forward Target Ratio in Continuous Tests')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # X坐标分布直方图
    ax2.hist(all_targets_x, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Forward Threshold')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('X Coordinate Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("consistency_test_visualization.png", dpi=300, bbox_inches='tight')
    print("Consistency test visualization saved to: consistency_test_visualization.png")
    plt.show()
    
    print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    test_curriculum_targets()