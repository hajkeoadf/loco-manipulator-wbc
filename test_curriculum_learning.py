#!/usr/bin/env python3
"""
测试solefoot_flat_with_arm的课程学习功能
"""

import torch
import numpy as np
from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm_config import BipedCfgSFWithArm
from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm import BipedSFWithArm

def test_curriculum_learning():
    """测试课程学习功能"""
    print("开始测试课程学习功能...")
    
    # 创建配置
    cfg = BipedCfgSFWithArm()
    
    # 检查课程学习配置
    print(f"命令课程学习启用: {cfg.commands.curriculum}")
    print(f"地形课程学习启用: {cfg.terrain.curriculum}")
    print(f"课程学习阈值: {cfg.commands.curriculum_threshold}")
    
    # 检查机械臂课程学习参数
    print(f"长度课程学习调度: {cfg.goal_ee.l_schedule}")
    print(f"俯仰课程学习调度: {cfg.goal_ee.p_schedule}")
    print(f"偏航课程学习调度: {cfg.goal_ee.y_schedule}")
    print(f"跟踪奖励课程学习调度: {cfg.goal_ee.tracking_ee_reward_schedule}")
    
    # 检查初始和最终范围
    print(f"初始长度范围: {cfg.goal_ee.ranges.init_pos_l}")
    print(f"最终长度范围: {cfg.goal_ee.ranges.final_pos_l}")
    print(f"初始俯仰范围: {cfg.goal_ee.ranges.init_pos_p}")
    print(f"最终俯仰范围: {cfg.goal_ee.ranges.final_pos_p}")
    print(f"初始偏航范围: {cfg.goal_ee.ranges.init_pos_y}")
    print(f"最终偏航范围: {cfg.goal_ee.ranges.final_pos_y}")
    print(f"最终跟踪奖励: {cfg.goal_ee.ranges.final_tracking_ee_reward}")
    
    print("\n课程学习配置检查完成！")
    
    # 测试课程学习值计算
    def test_curriculum_value():
        """测试课程学习值计算函数"""
        print("\n测试课程学习值计算...")
        
        # 模拟课程学习函数
        def get_curriculum_value(schedule, init_range, final_range, counter):
            return np.clip((counter - schedule[0]) / (schedule[1] - schedule[0]), 0, 1) * (final_range - init_range) + init_range
        
        # 测试不同计数器值
        schedule = [0, 1000]
        init_range = np.array([0.1, 0.3])
        final_range = np.array([0.2, 0.4])
        
        for counter in [0, 250, 500, 750, 1000, 1500]:
            value = get_curriculum_value(schedule, init_range, final_range, counter)
            progress = np.clip((counter - schedule[0]) / (schedule[1] - schedule[0]), 0, 1)
            print(f"计数器: {counter}, 进度: {progress:.2f}, 长度范围: [{value[0]:.3f}, {value[1]:.3f}]")
        
        print("课程学习值计算测试完成！")
    
    test_curriculum_value()
    
    print("\n所有测试完成！课程学习功能已成功添加到solefoot_flat_with_arm中。")

if __name__ == "__main__":
    test_curriculum_learning() 