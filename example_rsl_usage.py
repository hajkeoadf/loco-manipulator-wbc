#!/usr/bin/env python3
"""
RSL算法使用示例
展示如何使用迁移的rsl_rl算法进行训练
"""

import torch
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

from legged_gym.algorithm.rsl_ppo import RSLPPO
from legged_gym.algorithm.rsl_actor_critic import ActorCritic
from legged_gym.envs.solefoot_flat.solefoot_flat_yuxin import BipedSF


def create_rsl_algorithm():
    """创建RSL算法实例"""
    
    # 算法参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 观测维度（根据你的环境调整）
    num_actor_obs = 241  # 54 + 187 (height samples)
    num_critic_obs = 241
    num_actions = 14  # 8 legs + 6 arm
    
    # 创建Actor-Critic网络
    actor_critic = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        priv_encoder_dims=[64, 20],
        activation='elu',
        init_std=1.0,
        num_hist=10,
        num_prop=48
    )
    
    # 创建RSL PPO算法
    rsl_ppo = RSLPPO(
        actor_critic=actor_critic,
        num_learning_epochs=4,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device=device,
        mixing_schedule=[0.5, 2000, 4000],
        torque_supervision=True,
        torque_supervision_schedule=[0.1, 1000, 1000],
        adaptive_arm_gains=True,
        min_policy_std=None,
        dagger_update_freq=20,
        priv_reg_coef_schedule=[0, 0, 0]
    )
    
    return rsl_ppo


def train_with_rsl():
    """使用RSL算法进行训练"""
    
    print("=== RSL算法训练示例 ===")
    
    # 创建算法
    rsl_ppo = create_rsl_algorithm()
    
    # 初始化存储
    num_envs = 4096
    num_transitions_per_env = 24
    obs_shape = (241,)  # 根据你的环境调整
    critic_obs_shape = (241,)
    action_shape = (14,)
    
    rsl_ppo.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=num_transitions_per_env,
        actor_obs_shape=obs_shape,
        critic_obs_shape=critic_obs_shape,
        action_shape=action_shape
    )
    
    # 设置机械臂默认参数（如果使用自适应增益）
    default_arm_p_gains = torch.tensor([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
    default_arm_d_gains = torch.tensor([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    default_arm_dof_pos = torch.zeros(6)
    
    rsl_ppo.set_arm_default_coeffs(
        default_arm_p_gains=default_arm_p_gains,
        default_arm_d_gains=default_arm_d_gains,
        default_arm_dof_pos=default_arm_dof_pos
    )
    
    print("✅ RSL算法初始化完成")
    print(f"设备: {rsl_ppo.device}")
    print(f"环境数量: {num_envs}")
    print(f"每次转换数量: {num_transitions_per_env}")
    
    return rsl_ppo


def simulate_training_step(rsl_ppo):
    """模拟一个训练步骤"""
    
    print("\n=== 模拟训练步骤 ===")
    
    # 模拟观测数据
    num_envs = 4096
    device = rsl_ppo.device
    
    # 创建模拟观测
    obs = torch.randn(num_envs, 241, device=device)
    critic_obs = torch.randn(num_envs, 241, device=device)
    
    # 获取动作
    actions, actions_log_prob, values, adaptive_gains = rsl_ppo.act(obs, critic_obs)
    
    print(f"动作形状: {actions.shape}")
    print(f"动作对数概率形状: {actions_log_prob.shape}")
    print(f"价值形状: {values.shape}")
    if adaptive_gains is not None:
        print(f"自适应增益形状: {adaptive_gains.shape}")
    
    # 模拟环境步
    leg_rewards = torch.randn(num_envs, 1, device=device)
    arm_rewards = torch.randn(num_envs, 1, device=device)
    dones = torch.zeros(num_envs, 1, device=device, dtype=torch.bool)
    
    # 模拟信息字典
    infos = {
        'actions': actions,
        'values': values,
        'actions_log_prob': actions_log_prob,
        'action_mean': actions,
        'action_sigma': torch.ones_like(actions),
        'target_arm_torques': torch.randn(num_envs, 6, device=device),
        'current_arm_dof_pos': torch.randn(num_envs, 6, device=device),
        'current_arm_dof_vel': torch.randn(num_envs, 6, device=device)
    }
    
    # 处理环境步
    rsl_ppo.process_env_step(leg_rewards, arm_rewards, dones, infos)
    
    print("✅ 环境步处理完成")
    
    # 计算回报
    last_critic_obs = torch.randn(num_envs, 241, device=device)
    rsl_ppo.compute_returns(last_critic_obs)
    
    print("✅ 回报计算完成")
    
    # 更新策略
    mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_torque_supervision_loss = rsl_ppo.update()
    
    print("✅ 策略更新完成")
    print(f"价值损失: {mean_value_loss:.4f}")
    print(f"代理损失: {mean_surrogate_loss:.4f}")
    print(f"熵损失: {mean_entropy_loss:.4f}")
    print(f"扭矩监督损失: {mean_torque_supervision_loss:.4f}")
    
    # 更新计数器
    rsl_ppo.update_counter()
    
    # 获取课程学习参数
    mixing_ratio = rsl_ppo.get_value_mixing_ratio()
    torque_weight = rsl_ppo.get_torque_supervision_weight()
    
    print(f"价值混合比例: {mixing_ratio:.4f}")
    print(f"扭矩监督权重: {torque_weight:.4f}")


def compare_algorithms():
    """对比两种算法"""
    
    print("\n=== 算法对比 ===")
    
    # 创建RSL算法
    rsl_ppo = create_rsl_algorithm()
    
    # 对比特性
    comparison = {
        "历史编码器": "✅ CNN历史编码器",
        "分离控制头": "✅ 腿部和机械臂分离",
        "自适应增益": "✅ 机械臂自适应增益",
        "扭矩监督": "✅ 扭矩监督损失",
        "课程学习": "✅ 动态参数调整",
        "分离奖励": "✅ 腿部和机械臂独立奖励",
        "特权信息": "✅ 特权信息编码"
    }
    
    print("RSL算法特性:")
    for feature, status in comparison.items():
        print(f"  {feature}: {status}")
    
    print("\n与基础PPO相比的优势:")
    print("  1. 更好的历史信息利用")
    print("  2. 更精确的机械臂控制")
    print("  3. 更强的泛化能力")
    print("  4. 更稳定的训练过程")
    print("  5. 更好的任务适应性")


if __name__ == "__main__":
    print("RSL算法迁移示例")
    print("=" * 50)
    
    try:
        # 创建算法
        rsl_ppo = train_with_rsl()
        
        # 模拟训练步骤
        simulate_training_step(rsl_ppo)
        
        # 对比算法
        compare_algorithms()
        
        print("\n✅ 所有示例运行成功!")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc() 