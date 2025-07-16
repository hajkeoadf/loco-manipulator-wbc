# 双足机器人行走修复总结

## 问题分析

通过对比 `airbot` 成功实现和当前的 `solefoot_flat` 实现，发现了以下关键问题：

1. **观测空间不匹配**：观测空间定义不正确，缺少必要的高度测量
2. **奖励函数权重不合理**：一些关键奖励权重过低或过高，导致训练不稳定
3. **控制参数需要调整**：PD增益和动作缩放需要优化
4. **缺少重要的奖励函数**：缺少 `feet_air_time`、`stumble` 等关键奖励
5. **过于复杂的控制逻辑**：双足和机械臂分离控制增加了复杂性

## 主要修改

### 1. 配置文件修改 (`solefoot_flat_config.py`)

#### 环境参数
- 减少环境数量：`8192` → `4096` 以提高稳定性
- 修正动作数量：`8+6` → `14` 与实际关节数量匹配
- 修正观测空间：`36+18` → `57` 参考airbot
- 增加高度采样点：`117` → `187`

#### 地形参数
- 增加摩擦力：`0.4` → `1.0`
- 减少弹性：`0.8` → `0.0`
- 启用高度测量：`measure_heights = True`
- 使用airbot的测量点配置

#### 命令参数
- 简化命令数量：`5` → `4` (lin_vel_x, lin_vel_y, ang_vel_yaw, heading)
- 调整速度范围：参考airbot的成功配置
- 增加重采样时间：`5.0` → `10.0`

#### 控制参数
- 调整PD增益：参考airbot的刚度 `10` 和阻尼 `0.2/10`
- 减少decimation：`8` → `4` 提高控制频率
- 简化初始关节角度

#### 奖励权重
- 参考airbot的成功奖励权重
- 启用 `only_positive_rewards = True`
- 移除可能导致不稳定的复杂奖励
- 降低机械臂稳定性奖励权重

#### 域随机化
- 暂时关闭大部分随机化以提高稳定性
- 关闭摩擦、质量、推力等随机化

### 2. 主要实现修改 (`solefoot_flat.py`)

#### 观测计算
- 简化观测结构，参考airbot的实现
- 添加高度测量到观测空间
- 修正观测缓冲区初始化

#### 力矩计算
- 简化控制逻辑，移除双足和机械臂分离控制
- 统一使用PD控制器
- 移除复杂的动作延迟和FIFO机制

#### 奖励函数
- 添加缺失的 `_reward_feet_air_time()` 函数
- 添加缺失的 `_reward_stumble()` 函数
- 修正 `_reward_stand_still()` 函数
- 修正 `_reward_feet_contact_forces()` 函数
- 简化 `_reward_arm_stability()` 函数

#### 关节识别
- 简化关节识别逻辑，不再区分双足和机械臂
- 统一处理所有关节
- 修正动作空间大小

#### 步进函数
- 简化 `step()` 函数，移除复杂的动作处理
- 修正返回值格式
- 移除不必要的预处理步骤

## 最新修复内容（2024年更新）

### 3. 命令空间维度修复

#### 问题描述
训练时出现命令空间维度不匹配错误：
- `self.commands` 是5维，但 `commands_scale` 是4维
- 在 `step()` 方法中访问 `self.commands[:, :4]` 导致维度不匹配

#### 解决方案
1. **修正 `commands_scale` 维度**：
   ```python
   # 在 solefoot_flat.py 中
   self.commands_scale = torch.tensor(
       [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, 1.0, 1.0],  # 5维
       device=self.device,
       requires_grad=False,
   )
   ```

2. **修正 `step()` 方法返回值**：
   ```python
   # 修正前
   self.commands[:, :4] * self.commands_scale  # 4维
   
   # 修正后
   self.commands[:, :5] * self.commands_scale  # 5维命令输出
   ```

3. **恢复 `stand_still` 相关逻辑**：
   - 保持命令空间为5维：`[lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stand_still]`
   - 确保所有相关代码都使用5维命令空间

### 4. 观测历史维度修复

#### 问题描述
MLP_Encoder 输入维度不匹配：
- `obs_history` 维度是427，但MLP_Encoder期望240维
- 实际观测维度是235（48基础 + 187高度测量），但 `obs_history` 初始化错误

#### 解决方案
1. **修正MLP_Encoder配置**：
   ```python
   # 在 solefoot_flat_config.py 中
   class MLP_Encoder:
       num_input_dim = (BipedCfgSF.env.num_observations + BipedCfgSF.env.num_height_samples) * BipedCfgSF.env.obs_history_length  # (48 + 187) * 5 = 1175
   ```

2. **修正 `obs_history` 初始化**：
   ```python
   # 在 compute_observations 方法中
   actual_obs_dim = self.obs_buf.shape[1]  # 235 (48 + 187)
   if self.obs_history.shape[1] != actual_obs_dim * self.obs_history_length:
       self.obs_history = torch.zeros(
           self.num_envs,
           actual_obs_dim * self.obs_history_length,  # 235 * 5 = 1175
           device=self.device,
           dtype=torch.float,
       )
   ```

3. **修正 `reset_idx` 方法**：
   ```python
   # 使用完整的观测（包含高度测量）
   if self.cfg.terrain.measure_heights:
       self.measured_heights = self._get_heights()
   self.compute_observations()
   self.obs_history[env_ids] = self.obs_buf[env_ids].repeat(1, self.obs_history_length)
   ```

### 5. Actor网络输入维度修复

#### 问题描述
Actor网络输入维度不匹配：
- Actor网络期望56维输入，但实际输入是243维
- `self.num_obs` 设置为48，但实际观测维度是235

#### 解决方案
1. **修正 `self.num_obs` 计算**：
   ```python
   # 在 _init_buffers 方法中
   self.num_obs = self.cfg.env.num_observations + self.cfg.env.num_height_samples  # 48 + 187 = 235
   ```

2. **确保Actor网络输入维度正确**：
   - Actor网络输入维度 = `self.num_obs` (235) + `encoder.num_output_dim` (3) + `self.env.num_commands` (5) = 243

### 6. 高度测量修复

#### 问题描述
高度测量维度不匹配：
- `self.obs_buf` 维度是49，但噪声向量维度是235
- 在 `reset_idx` 中调用 `compute_observations()` 时，`self.measured_heights` 未正确初始化

#### 解决方案
1. **在 `reset_idx` 中先计算高度测量**：
   ```python
   # 先计算高度测量
   if self.cfg.terrain.measure_heights:
       self.measured_heights = self._get_heights()
   # 再使用完整的观测
   self.compute_observations()
   ```

## 关键改进点

### 1. 观测空间优化
```python
# 参考airbot的观测结构
obs_buf = torch.cat((
    self.base_ang_vel * self.obs_scales.ang_vel,
    self.projected_gravity,
    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    self.dof_vel * self.obs_scales.dof_vel,
    self.actions,
), dim=-1)
```

### 2. 简化控制逻辑
```python
# 统一的PD控制器
torques = (
    self.p_gains * 
    (actions_scaled + self.default_dof_pos - self.dof_pos)
    - self.d_gains * self.dof_vel
)
```

### 3. 关键奖励函数
```python
# 步态奖励
def _reward_feet_air_time(self):
    contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.
    contact_filt = torch.logical_or(contact, self.last_contacts)
    # ... 计算空中时间奖励

# 防绊倒奖励
def _reward_stumble(self):
    return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5., dim=1)
```

### 4. 维度一致性保证
```python
# 确保所有维度一致
self.num_obs = 235  # 48基础 + 187高度测量
self.commands_scale = 5维  # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stand_still]
self.obs_history = 1175维  # 235 * 5历史长度
```

## 预期效果

1. **稳定性提升**：通过参考airbot的成功配置，提高训练稳定性
2. **行走能力改善**：通过正确的奖励函数和观测空间，改善行走性能
3. **训练效率提高**：简化控制逻辑，减少不必要的复杂性
4. **更好的泛化性**：通过合理的域随机化设置，提高泛化能力
5. **维度一致性**：确保所有张量维度匹配，避免运行时错误

## 测试建议

1. 运行测试脚本验证环境正常工作
2. 进行短期训练测试，观察奖励曲线
3. 逐步启用域随机化，观察性能变化
4. 调整奖励权重，找到最佳平衡
5. 验证所有维度匹配，确保训练稳定

## 注意事项

1. 这些修改主要参考了airbot的成功实现
2. 暂时关闭了一些可能导致不稳定的功能
3. 需要在训练过程中逐步调优参数
4. 建议先在小规模环境中测试，再扩展到大规模训练
5. 特别注意维度一致性，确保所有张量操作正确
6. 保持命令空间为5维，包含 `stand_still` 命令