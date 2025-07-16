#!/usr/bin/env python3

def test_config_file_content():
    """直接检查配置文件内容"""
    print("=== 直接检查配置文件内容 ===")
    
    config_file = "legged_gym/envs/solefoot_flat/solefoot_flat_config.py"
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        print("1. 检查动作空间维度...")
        if "num_actions = 8 + 6" in content:
            print("   ✓ 动作空间维度配置正确: 8 + 6")
        elif "num_actions = 14" in content:
            print("   ✓ 动作空间维度配置正确: 14")
        else:
            print("   ✗ 动作空间维度配置不正确")
            # 查找实际的配置
            import re
            match = re.search(r'num_actions\s*=\s*(\d+)', content)
            if match:
                print(f"   实际配置: num_actions = {match.group(1)}")
        
        print("\n2. 检查观测空间维度...")
        if "num_observations = 36 + 18" in content:
            print("   ✓ 观测空间维度配置正确: 36 + 18")
        elif "num_observations = 54" in content:
            print("   ✓ 观测空间维度配置正确: 54")
        else:
            print("   ✗ 观测空间维度配置不正确")
            # 查找实际的配置
            import re
            match = re.search(r'num_observations\s*=\s*(\d+)', content)
            if match:
                print(f"   实际配置: num_observations = {match.group(1)}")
        
        print("\n3. 检查机械臂关节配置...")
        arm_joints = ["J1", "J2", "J3", "J4", "J5", "J6"]
        for joint in arm_joints:
            if f'"{joint}": 50' in content:
                print(f"   ✓ {joint}: stiffness=50")
            elif f'"{joint}": 40' in content:
                print(f"   ⚠ {joint}: stiffness=40 (较低)")
            elif f'"{joint}": 30' in content:
                print(f"   ⚠ {joint}: stiffness=30 (较低)")
            else:
                print(f"   ✗ {joint}: 未找到刚度配置")
        
        print("\n4. 检查机械臂阻尼配置...")
        for joint in arm_joints:
            if f'"{joint}": 1.5' in content:
                print(f"   ✓ {joint}: damping=1.5")
            elif f'"{joint}": 1.2' in content:
                print(f"   ⚠ {joint}: damping=1.2 (较低)")
            elif f'"{joint}": 1.0' in content:
                print(f"   ⚠ {joint}: damping=1.0 (较低)")
            else:
                print(f"   ✗ {joint}: 未找到阻尼配置")
        
        print("\n5. 检查机械臂默认角度...")
        if '"J2": -0.785' in content:
            print("   ✓ J2默认角度配置正确: -0.785")
        elif '"J2": -1.57' in content:
            print("   ⚠ J2默认角度配置: -1.57 (旧值)")
        else:
            print("   ✗ J2默认角度配置不正确")
        
        print("\n6. 检查机械臂稳定性奖励...")
        if "arm_stability = 1.0" in content:
            print("   ✓ 机械臂稳定性奖励配置正确: 1.0")
        elif "arm_stability = 0.5" in content:
            print("   ⚠ 机械臂稳定性奖励配置: 0.5 (较低)")
        else:
            print("   ✗ 机械臂稳定性奖励配置不正确")
        
        print("\n7. 检查双足关节配置...")
        leg_joints = ["abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint"]
        for joint in leg_joints:
            if f'"{joint}": 45' in content:
                print(f"   ✓ {joint}: stiffness=45")
            else:
                print(f"   ✗ {joint}: 未找到刚度配置")
        
    except FileNotFoundError:
        print(f"✗ 配置文件未找到: {config_file}")
    except Exception as e:
        print(f"✗ 读取配置文件时出错: {e}")

def test_urdf_file():
    """检查URDF文件"""
    print("\n=== 检查URDF文件 ===")
    
    urdf_file = "resources/robots/SF_TRON1A/urdf/robot.urdf"
    
    try:
        with open(urdf_file, 'r') as f:
            content = f.read()
        
        print("1. 检查机械臂关节定义...")
        arm_joints = ["J1", "J2", "J3", "J4", "J5", "J6"]
        for joint in arm_joints:
            if f'<joint name="{joint}"' in content:
                print(f"   ✓ {joint} 关节已定义")
            else:
                print(f"   ✗ {joint} 关节未定义")
        
        print("\n2. 检查双足关节定义...")
        leg_joints = ["abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint"]
        for joint in leg_joints:
            if f'<joint name="{joint}"' in content:
                print(f"   ✓ {joint} 关节已定义")
            else:
                print(f"   ✗ {joint} 关节未定义")
        
        print("\n3. 检查transmission配置...")
        for joint in arm_joints:
            if f'<transmission name="{joint}_tran"' in content:
                print(f"   ✓ {joint} transmission已配置")
            else:
                print(f"   ✗ {joint} transmission未配置")
        
    except FileNotFoundError:
        print(f"✗ URDF文件未找到: {urdf_file}")
    except Exception as e:
        print(f"✗ 读取URDF文件时出错: {e}")

if __name__ == "__main__":
    test_config_file_content()
    test_urdf_file()
    print("\n=== 测试完成 ===") 