#!/usr/bin/env python3

import torch
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

from legged_gym.algorithm.rsl_actor_critic import ActorCritic

def test_actor_critic():
    """测试修改后的ActorCritic类"""
    
    # 配置参数（基于solefoot_flat_with_arm_config.py）
    num_actor_obs = 912  # 总观察维度：基础(57) + 机械臂(6) + 高度(187) + 特权(4) + 历史(650)
    num_critic_obs = 912
    num_actions = 14  # 8个腿部关节 + 6个机械臂关节
    num_hist = 10
    num_prop = 65  # 本体感受观察维度
    
    print(f"测试参数:")
    print(f"  num_actor_obs: {num_actor_obs}")
    print(f"  num_critic_obs: {num_critic_obs}")
    print(f"  num_actions: {num_actions}")
    print(f"  num_hist: {num_hist}")
    print(f"  num_prop: {num_prop}")
    
    # 创建ActorCritic实例
    actor_critic = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[64, 20],
        activation='elu',
        init_std=1.0,
        num_hist=num_hist,
        num_prop=num_prop
    )
    
    print(f"\nActorCritic创建成功!")
    
    # 创建测试观察数据
    batch_size = 4
    observations = torch.randn(batch_size, num_actor_obs)
    critic_observations = torch.randn(batch_size, num_critic_obs)
    
    print(f"\n测试观察数据形状:")
    print(f"  observations: {observations.shape}")
    print(f"  critic_observations: {critic_observations.shape}")
    
    # 测试特权观察编码
    print(f"\n测试特权观察编码...")
    try:
        # 测试infer_priv_latent
        priv_latent = actor_critic.actor.infer_priv_latent(observations)
        print(f"  特权观察编码成功! 输出形状: {priv_latent.shape}")
        
        # 验证特权观察提取
        base_obs_dim = 57
        arm_obs_dim = 6
        height_obs_dim = 187
        priv_start = base_obs_dim + arm_obs_dim + height_obs_dim
        priv_end = priv_start + 4
        extracted_priv = observations[:, priv_start:priv_end]
        print(f"  提取的特权观察形状: {extracted_priv.shape}")
        
    except Exception as e:
        print(f"  特权观察编码失败: {e}")
        return False
    
    # 测试历史观察编码
    print(f"\n测试历史观察编码...")
    try:
        # 测试infer_hist_latent
        hist_latent = actor_critic.actor.infer_hist_latent(observations)
        print(f"  历史观察编码成功! 输出形状: {hist_latent.shape}")
        
        # 验证历史观察提取
        hist_start = base_obs_dim + arm_obs_dim + height_obs_dim + 4
        hist_end = hist_start + num_hist * num_prop
        extracted_hist = observations[:, hist_start:hist_end]
        print(f"  提取的历史观察形状: {extracted_hist.shape}")
        print(f"  重塑后的历史观察形状: {extracted_hist.reshape(-1, num_hist, num_prop).shape}")
        
    except Exception as e:
        print(f"  历史观察编码失败: {e}")
        return False
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    try:
        # 测试特权编码模式
        actions_priv, adaptive_gains_priv = actor_critic.actor(observations, hist_encoding=False)
        print(f"  特权编码模式成功!")
        print(f"    动作形状: {actions_priv.shape}")
        print(f"    自适应增益形状: {adaptive_gains_priv.shape if adaptive_gains_priv is not None else 'None'}")
        
        # 测试历史编码模式
        actions_hist, adaptive_gains_hist = actor_critic.actor(observations, hist_encoding=True)
        print(f"  历史编码模式成功!")
        print(f"    动作形状: {actions_hist.shape}")
        print(f"    自适应增益形状: {adaptive_gains_hist.shape if adaptive_gains_hist is not None else 'None'}")
        
    except Exception as e:
        print(f"  前向传播失败: {e}")
        return False
    
    # 测试Critic
    print(f"\n测试Critic...")
    try:
        values = actor_critic.critic(critic_observations)
        print(f"  Critic前向传播成功! 输出形状: {values.shape}")
        
    except Exception as e:
        print(f"  Critic前向传播失败: {e}")
        return False
    
    # 测试完整的act方法
    print(f"\n测试完整的act方法...")
    try:
        actions, actions_log_prob, values, adaptive_gains = actor_critic.act(
            observations, critic_observations, hist_encoding=False
        )
        print(f"  act方法成功!")
        print(f"    动作形状: {actions.shape}")
        print(f"    动作对数概率形状: {actions_log_prob.shape}")
        print(f"    价值形状: {values.shape}")
        print(f"    自适应增益形状: {adaptive_gains.shape if adaptive_gains is not None else 'None'}")
        
    except Exception as e:
        print(f"  act方法失败: {e}")
        return False
    
    print(f"\n✅ 所有测试通过!")
    return True

if __name__ == "__main__":
    success = test_actor_critic()
    if success:
        print("\n🎉 ActorCritic修改成功!")
    else:
        print("\n❌ ActorCritic修改存在问题!")
        sys.exit(1) 