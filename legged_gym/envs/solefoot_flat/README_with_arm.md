# Solefoot Flat with Arm Environment

这是一个基于`solefoot_flat`的双足机器人+机械臂复合环境，专门用于训练双足机器人（SF_TRON1A）配合机械臂的强化学习任务。

## 环境特性

### 机器人系统
- **双足机器人（SF_TRON1A）**: 8个关节（4个腿部关节 × 2条腿）
- **机械臂**: 6个关节（J1-J6）
- **总关节数**: 14个（8个腿部 + 6个机械臂）

### 观察空间
- **基础观察**: 54维（来自父类）
- **机械臂观察**: 6维（目标位置 + 目标姿态）
- **总观察维度**: 60维

### 动作空间
- **腿部动作**: 14维（8个腿部关节 + 6个其他关节）
- **机械臂动作**: 6维（6个机械臂关节）
- **总动作维度**: 20维

### 奖励函数
- **腿部奖励**: 继承自父类，包括速度跟踪、姿态控制等
- **机械臂奖励**:
  - `tracking_ee_cart`: 末端执行器位置跟踪（笛卡尔坐标系）
  - `tracking_ee_sphere`: 末端执行器位置跟踪（球坐标系）
  - `tracking_ee_orn`: 末端执行器姿态跟踪
  - `arm_energy_abs_sum`: 机械臂能量消耗

## 文件结构

```
solefoot_flat/
├── solefoot_flat_with_arm.py          # 主环境类
├── solefoot_flat_with_arm_config.py   # 配置文件
├── solefoot_flat.py                   # 父类环境
├── solefoot_flat_config.py            # 父类配置
└── README_with_arm.md                 # 本文件
```

## 使用方法

### 1. 基本使用

```python
from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm import BipedSFWithArm
from legged_gym.envs.solefoot_flat.solefoot_flat_with_arm_config import BipedCfgSFWithArm

# 创建配置
cfg = BipedCfgSFWithArm()

# 创建环境
env = BipedSFWithArm(
    cfg=cfg,
    sim_params=None,
    physics_engine=None,
    sim_device='cuda:0',
    headless=False
)

# 重置环境
obs = env.reset()

# 执行动作
actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
obs, privileged_obs, rewards, dones, extras = env.step(actions)
```

### 2. 与RSL算法结合

```python
from legged_gym.algorithm.rsl_actor_critic import ActorCritic

# 创建Actor-Critic网络
actor_critic = ActorCritic(
    num_actor_obs=env.get_num_actor_obs(),
    num_critic_obs=env.get_num_critic_obs(),
    num_actions=env.get_num_actions_total(),
    num_leg_actions=env.get_num_leg_actions(),
    num_arm_actions=env.get_num_arm_actions(),
    adaptive_arm_gains=env.get_adaptive_arm_gains(),
    adaptive_arm_gains_scale=env.get_adaptive_arm_gains_scale(),
    leg_control_head_hidden_dims=env.get_leg_control_head_hidden_dims(),
    arm_control_head_hidden_dims=env.get_arm_control_head_hidden_dims(),
    # ... 其他参数
)
```

### 3. 训练脚本

使用提供的训练脚本：

```bash
cd legged_gym/scripts
python train_solefoot_with_arm.py
```

### 4. 测试脚本

使用提供的测试脚本验证环境：

```bash
cd legged_gym/scripts
python test_solefoot_with_arm.py
```

## 配置说明

### 主要配置参数

#### 环境配置
- `num_envs`: 并行环境数量（默认4096）
- `num_observations`: 观察维度（60）
- `num_actions`: 动作维度（20）
- `episode_length_s`: 回合长度（20秒）

#### 机械臂配置
- `goal_ee.command_mode`: 目标模式（'cart'或'sphere'）
- `goal_ee.traj_time`: 轨迹时间范围
- `goal_ee.hold_time`: 保持时间范围
- `arm.osc_kp`: 操作空间控制Kp增益
- `arm.osc_kd`: 操作空间控制Kd增益

#### 奖励配置
- `rewards.scales.tracking_ee_cart`: 位置跟踪奖励权重
- `rewards.scales.tracking_ee_sphere`: 球坐标系跟踪奖励权重
- `rewards.scales.tracking_ee_orn`: 姿态跟踪奖励权重
- `rewards.scales.arm_energy_abs_sum`: 机械臂能量奖励权重

## 关键方法

### 环境接口
- `step(actions)`: 执行动作并返回结果
- `reset()`: 重置环境
- `get_observations()`: 获取观察
- `get_rewards()`: 获取奖励
- `get_dones()`: 获取完成标志

### RSL算法接口
- `get_num_leg_actions()`: 获取腿部动作数量
- `get_num_arm_actions()`: 获取机械臂动作数量
- `get_num_actions_total()`: 获取总动作数量
- `get_num_actor_obs()`: 获取Actor观察数量
- `get_num_critic_obs()`: 获取Critic观察数量

### 机械臂控制
- `get_arm_ee_control_torques()`: 获取操作空间控制力矩
- `update_curr_ee_goal()`: 更新当前末端执行器目标
- `_resample_ee_goal()`: 重新采样末端执行器目标

## 注意事项

1. **URDF文件**: 确保URDF文件路径正确，包含机械臂关节定义
2. **关节顺序**: 机械臂关节顺序为J1-J6，确保与URDF文件一致
3. **观察维度**: 观察空间包含基础观察和机械臂观察，总维度为60
4. **动作维度**: 动作空间包含腿部和机械臂动作，总维度为20
5. **奖励平衡**: 需要平衡腿部和机械臂的奖励权重

## 故障排除

### 常见问题

1. **维度不匹配**: 检查观察和动作维度是否正确
2. **URDF加载失败**: 检查URDF文件路径和格式
3. **机械臂关节未找到**: 检查URDF文件中的关节名称
4. **奖励计算错误**: 检查奖励函数的输入维度

### 调试建议

1. 使用测试脚本验证环境基本功能
2. 检查观察和动作的维度匹配
3. 验证URDF文件的正确性
4. 监控奖励函数的输出

## 扩展功能

### 自定义奖励函数
可以通过继承`BipedSFWithArm`类并重写奖励函数来添加自定义奖励：

```python
class CustomBipedSFWithArm(BipedSFWithArm):
    def _reward_custom(self):
        # 自定义奖励计算
        return custom_reward
```

### 自定义目标生成
可以重写目标生成方法来改变末端执行器的目标生成策略：

```python
def _resample_ee_goal(self, env_ids, is_init=False):
    # 自定义目标生成逻辑
    pass
```

### 自定义控制策略
可以重写控制方法来改变机械臂的控制策略：

```python
def _compute_torques(self, actions):
    # 自定义力矩计算
    pass
``` 