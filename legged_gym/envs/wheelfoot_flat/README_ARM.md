# 机械臂子类使用说明

## 概述

`BipedSFArm` 是一个继承自 `BipedSF` 的机械臂子类，专门用于处理双足+机械臂的机器人。该类提供了完整的机械臂控制、观测和奖励功能。

## 文件结构

```
solefoot_flat/
├── solefoot_flat_arm.py          # 机械臂子类实现
├── solefoot_flat_arm_config.py   # 机械臂配置文件
├── test_arm_class.py             # 测试脚本
└── README_ARM.md                 # 本说明文档
```

## 主要特性

### 1. 关节分类
- **腿部关节**: 8个 (abad_L_Joint, hip_L_Joint, knee_L_Joint, ankle_L_Joint, abad_R_Joint, hip_R_Joint, knee_R_Joint, ankle_R_Joint)
- **机械臂关节**: 6个 (J1, J2, J3, J4, J5, J6)
- **总关节数**: 14个

### 2. 观测空间
- **基础观测**: 54维 (包含腿部状态、IMU、命令等)
- **机械臂观测**: 31维 (关节状态 + 末端状态 + 目标状态)
- **总观测维度**: 85维

### 3. 动作空间
- **腿部动作**: 8维
- **机械臂动作**: 6维
- **总动作维度**: 14维

## 核心方法

### 观测方法

```python
# 获取腿部观测
leg_obs = env.get_leg_obs()
# 返回: 腿部关节位置、速度、力矩 + 足端状态

# 获取机械臂观测
arm_obs = env.get_arm_obs()
# 返回: 机械臂关节位置、速度、力矩 + 末端执行器状态 + 目标状态
```

### 控制方法

```python
# 设置机械臂目标
env.set_arm_target(env_ids, target_pos, target_vel=None)

# 获取机械臂状态
arm_pos, arm_vel = env.get_arm_state(env_ids)

# 获取末端执行器状态
ee_pos, ee_orn, ee_vel = env.get_ee_state(env_ids)
```

### 运动学方法

```python
# 逆运动学
joint_pos = env.inverse_kinematics(target_pos, target_orn=None, env_ids=None)

# 正运动学
ee_pos, ee_orn = env.forward_kinematics(joint_positions, env_ids=None)

# 雅可比矩阵
jacobian = env.compute_arm_jacobian(env_ids)
```

## 配置说明

### 环境配置 (`BipedCfgSFArm`)

```python
class env:
    num_observations = 85  # 基础观测 + 机械臂观测
    num_actions = 14       # 8腿 + 6臂
    ee_idx = 10           # 末端执行器索引
```

### 控制配置

```python
class control:
    stiffness = {
        # 腿部关节 - 较低刚度
        "abad_L_Joint": 10,
        # ...
        # 机械臂关节 - 较高刚度
        "J1": 100,
        "J2": 100,
        # ...
    }
    damping = {
        # 腿部关节 - 较低阻尼
        "abad_L_Joint": 0.2,
        # ...
        # 机械臂关节 - 较高阻尼
        "J1": 10,
        "J2": 10,
        # ...
    }
```

### 奖励配置

```python
class rewards:
    class scales:
        # 基础奖励
        tracking_lin_vel = 2.0
        orientation = -0.5
        # ...
        
        # 机械臂奖励
        arm_tracking = 1.0      # 机械臂跟踪奖励
        arm_energy = -0.01      # 机械臂能量惩罚
        arm_torques = -0.001    # 机械臂力矩惩罚
        arm_stability = -0.01   # 机械臂稳定性惩罚
```

## 使用方法

### 1. 基本使用

```python
from legged_gym.envs.solefoot_flat.solefoot_flat_arm import BipedSFArm
from legged_gym.envs.solefoot_flat.solefoot_flat_arm_config import BipedCfgSFArm

# 创建配置
cfg = BipedCfgSFArm()

# 创建环境
env = BipedSFArm(cfg, sim_params, physics_engine, sim_device, headless)

# 执行动作
actions = torch.randn(env.num_envs, env.num_dofs, device=env.device)
obs, rew, reset, extras, obs_history, commands, critic_obs = env.step(actions)
```

### 2. 机械臂控制

```python
# 设置机械臂目标
env_ids = torch.arange(10, device=env.device)
target_pos = torch.randn(len(env_ids), env.num_arm_dofs, device=env.device)
env.set_arm_target(env_ids, target_pos)

# 获取机械臂状态
arm_pos, arm_vel = env.get_arm_state(env_ids)
ee_pos, ee_orn, ee_vel = env.get_ee_state(env_ids)
```

### 3. 运动学计算

```python
# 逆运动学
target_pos = torch.randn(len(env_ids), 3, device=env.device)
joint_pos = env.inverse_kinematics(target_pos, env_ids=env_ids)

# 正运动学
joint_positions = torch.randn(len(env_ids), env.num_arm_dofs, device=env.device)
ee_pos, ee_orn = env.forward_kinematics(joint_positions, env_ids=env_ids)
```

## 训练配置

### RSL算法配置

```python
class BipedCfgPPOSFArm:
    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "RSLPPO"
        
    class policy:
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        
    class algorithm:
        learning_rate = 1.0e-3
        num_learning_epochs = 5
        num_mini_batches = 4
```

## 测试

运行测试脚本验证功能：

```bash
cd /home/ril/loco-manipulator-wbc/legged_gym/envs/solefoot_flat
python test_arm_class.py
```

## 注意事项

1. **关节顺序**: 腿部关节在前8个，机械臂关节在后6个
2. **观测维度**: 机械臂观测包含关节状态、末端状态和目标状态
3. **控制参数**: 机械臂使用更高的刚度和阻尼参数
4. **奖励设计**: 包含机械臂跟踪、能量、力矩和稳定性奖励
5. **运动学**: 当前提供简化版本，可根据需要扩展

## 扩展建议

1. **真实运动学**: 实现基于URDF的真实正逆运动学
2. **雅可比矩阵**: 基于机器人几何的真实雅可比矩阵计算
3. **任务奖励**: 根据具体任务设计更精细的奖励函数
4. **安全约束**: 添加关节限位、碰撞检测等安全约束
5. **多目标控制**: 支持腿部运动和机械臂操作的协调控制

## 与RSL算法的集成

该类完全兼容RSL算法框架：

- 支持 `RSLPPO` 算法
- 支持 `ActorCritic` 网络
- 支持历史编码器
- 支持特权观测训练

所有RSL开头的算法文件都可以直接使用该类进行训练。 