#!/usr/bin/env python3
"""
测试机械臂修复的脚本
验证配置文件、观测空间、动作空间等修改是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接读取配置文件，避免导入问题
def test_config():
    """测试配置文件修改"""
    print("=== 测试配置文件修改 ===")
    
    config_path = "legged_gym/envs/solefoot_flat/solefoot_flat_config.py"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # 测试动作数
    if "num_actions = 8 + 6" in config_content:
        print("✅ 动作数配置正确")
    else:
        print("❌ 动作数配置错误")
        return False
    
    # 测试观测数
    if "num_observations = 36 + 18" in config_content:
        print("✅ 观测数配置正确")
    else:
        print("❌ 观测数配置错误")
        return False
    
    # 测试机械臂刚度
    if all(f'"J{i}": 40' in config_content for i in range(1, 7)):
        print("✅ 机械臂刚度配置正确")
    else:
        print("❌ 机械臂刚度配置错误")
        return False
    
    # 测试机械臂阻尼
    if all(f'"J{i}": 1.2' in config_content for i in range(1, 7)):
        print("✅ 机械臂阻尼配置正确")
    else:
        print("❌ 机械臂阻尼配置错误")
        return False
    
    # 测试机械臂默认角度
    if '"J2": -1.57' in config_content:
        print("✅ 机械臂默认角度配置正确")
    else:
        print("❌ 机械臂默认角度配置错误")
        return False
    
    # 测试机械臂稳定性奖励权重
    if "arm_stability = -0.5" in config_content:
        print("✅ 机械臂稳定性奖励权重配置正确")
    else:
        print("❌ 机械臂稳定性奖励权重配置错误")
        return False
    
    return True

def test_urdf():
    """测试URDF文件"""
    print("\n=== 测试URDF文件 ===")
    
    urdf_path = "resources/robots/SF_TRON1A/urdf/robot.urdf"
    if not os.path.exists(urdf_path):
        print(f"❌ URDF文件不存在: {urdf_path}")
        return False
    
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # 检查机械臂关节
    arm_joints = [f"J{i}" for i in range(1, 7)]
    all_joints_found = True
    for joint in arm_joints:
        if joint in urdf_content:
            print(f"✅ 找到机械臂关节: {joint}")
        else:
            print(f"❌ 未找到机械臂关节: {joint}")
            all_joints_found = False
    
    # 检查transmission配置
    all_transmissions_found = True
    for joint in arm_joints:
        if f"{joint}_tran" in urdf_content:
            print(f"✅ 找到transmission配置: {joint}_tran")
        else:
            print(f"❌ 未找到transmission配置: {joint}_tran")
            all_transmissions_found = False
    
    return all_joints_found and all_transmissions_found

def test_python_files():
    """测试Python文件修改"""
    print("\n=== 测试Python文件修改 ===")
    
    python_path = "legged_gym/envs/solefoot_flat/solefoot_flat.py"
    if not os.path.exists(python_path):
        print(f"❌ Python文件不存在: {python_path}")
        return False
    
    with open(python_path, 'r') as f:
        python_content = f.read()
    
    # 检查机械臂索引
    if "self.arm_dof_indices = torch.tensor([8, 9, 10, 11, 12, 13]" in python_content:
        print("✅ 机械臂索引配置正确")
    else:
        print("❌ 机械臂索引配置错误")
        return False
    
    # 检查机械臂力矩计算函数
    if "def _compute_arm_torques(self, actions):" in python_content:
        print("✅ 机械臂力矩计算函数存在")
    else:
        print("❌ 机械臂力矩计算函数不存在")
        return False
    
    # 检查机械臂稳定性奖励函数
    if "def _reward_arm_stability(self):" in python_content:
        print("✅ 机械臂稳定性奖励函数存在")
    else:
        print("❌ 机械臂稳定性奖励函数不存在")
        return False
    
    return True

def main():
    """主测试函数"""
    print("开始测试机械臂修复...")
    
    try:
        config_ok = test_config()
        urdf_ok = test_urdf()
        python_ok = test_python_files()
        
        if config_ok and urdf_ok and python_ok:
            print("\n🎉 所有测试通过！机械臂修复成功。")
            return True
        else:
            print("\n❌ 部分测试失败")
            return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 