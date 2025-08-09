#!/usr/bin/env python3
"""
手动机械臂测试脚本 - 简化版
用于快速验证机械臂运动能力
"""

def quick_arm_test():
    """快速机械臂测试"""
    try:
        import torch
        from legged_gym.envs import *
        from legged_gym.utils import get_args, task_registry
        
        print("🚀 启动快速机械臂测试...")
        
        # 创建环境
        args = get_args()
        args.task = "solefoot_flat_with_arm"
        args.headless = True  # 无头模式，更快
        args.num_envs = 1  # 只用一个环境
        
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        print("✅ 环境创建成功")
        
        # 重置环境
        obs = env.reset()
        print(f"✅ 环境重置成功，观察维度: {obs.shape}")
        
        # 测试基本信息
        print(f"总动作数: {env.num_actions}")
        print(f"机械臂动作数: {env.get_num_arm_actions()}")
        print(f"腿部动作数: {env.get_num_leg_actions()}")
        
        # 执行一些测试步骤
        print("\n开始运动测试...")
        initial_ee_pos = None
        
        for step in range(50):
            # 生成测试动作
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            
            # 机械臂动作：简单的周期性运动
            for i in range(6):
                actions[0, 8+i] = 0.3 * torch.sin(step * 0.1 + i)
            
            # 执行步骤
            obs, rew, arm_rew, done, info, obs_history, critic_obs = env.step(actions)
            
            # 记录初始位置
            if step == 0:
                initial_ee_pos = env.ee_pos[0].clone()
                print(f"初始末端执行器位置: {initial_ee_pos.cpu().numpy()}")
            
            # 每10步输出一次
            if step % 10 == 0 and step > 0:
                current_ee_pos = env.ee_pos[0]
                movement = torch.norm(current_ee_pos - initial_ee_pos).item()
                print(f"步数 {step}: 末端执行器移动距离 {movement:.4f}m")
        
        # 最终结果
        final_ee_pos = env.ee_pos[0]
        total_movement = torch.norm(final_ee_pos - initial_ee_pos).item()
        
        print(f"\n📊 测试结果:")
        print(f"总移动距离: {total_movement:.4f}m")
        print(f"初始位置: {initial_ee_pos.cpu().numpy()}")
        print(f"最终位置: {final_ee_pos.cpu().numpy()}")
        
        if total_movement > 0.01:
            print("✅ 测试通过：机械臂能够运动！")
            return True
        else:
            print("❌ 测试失败：机械臂运动不足")
            return False
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_arm_test()
    if success:
        print("\n🎉 机械臂运动验证成功！")
        print("现在可以运行完整训练：python train_solefoot_with_arm.py")
    else:
        print("\n⚠️ 请检查配置和URDF文件")
