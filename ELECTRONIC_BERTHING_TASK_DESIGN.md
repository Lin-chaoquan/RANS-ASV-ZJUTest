# 电子泊船任务设计文档

## 任务概述

### **任务目标**
无人船电子泊船系统，要求在有限时间内快速驶入船位并保持在船位中心。

### **关键约束**
1. **红线（墙壁）**: 绝对禁止碰撞，碰撞即失败
2. **绿线（入口）**: 允许轻微接触，但不应过度依赖
3. **时间限制**: 快速完成泊船任务
4. **精度要求**: 精确保持在船位中心

## 奖励函数设计

### **1. 基础奖励权重**

| 参数 | 数值 | 说明 |
|------|------|------|
| `position_scale` | 40.0 | 位置奖励权重，高权重确保精确定位 |
| `heading_scale` | 25.0 | 航向奖励权重，高权重确保航向对齐 |
| `velocity_penalty_scale` | 0.5 | 速度惩罚权重，低权重允许快速运动 |
| `angular_velocity_penalty_scale` | 1.0 | 角速度惩罚权重，低权重允许快速转向 |

### **2. 泊船特定奖励**

| 参数 | 数值 | 说明 |
|------|------|------|
| `approach_reward_scale` | 20.0 | 接近奖励权重，鼓励快速接近 |
| `center_keeping_reward_scale` | 30.0 | 中心保持奖励权重，确保在中心 |
| `time_efficiency_scale` | 15.0 | 时间效率奖励权重，鼓励快速完成 |

### **3. 碰撞检测策略**

| 碰撞类型 | 惩罚值 | 策略 |
|----------|--------|------|
| **红线（墙壁）** | -500.0 | 极重惩罚，绝对禁止 |
| **绿线（入口）** | -5.0 | 轻微惩罚，允许接触 |

### **4. 成功检测标准**

| 参数 | 数值 | 说明 |
|------|------|------|
| `position_tolerance` | 0.1 | 位置容差，严格要求精确定位 |
| `heading_tolerance` | 0.05 | 航向容差，严格要求航向对齐 |
| `success_reward` | 300.0 | 成功奖励，高奖励强化成功行为 |

## 奖励函数逻辑

### **1. 渐进式位置奖励**

```python
def _compute_progressive_position_reward(self, position_error):
    """渐进式位置奖励：距离越近奖励越高，鼓励快速接近"""
    base_reward = self.position_scale * torch.exp(-position_error * 0.8)
    
    # 距离阈值和对应倍数
    distance_thresholds = [2.5, 1.5, 0.8, 0.3]
    reward_multipliers = [0.5, 1.0, 1.8, 3.0]
    
    # 根据距离应用倍数，更激进的接近奖励
    for threshold, multiplier in zip(distance_thresholds, reward_multipliers):
        mask = position_error < threshold
        reward_multiplier[mask] = multiplier
    
    return base_reward * reward_multiplier
```

**设计思路**: 
- 距离越近，奖励倍数越高
- 鼓励船体快速接近目标
- 在接近目标时给予更高奖励

### **2. 中心保持奖励**

```python
def _compute_center_keeping_reward(self, position_error):
    """中心保持奖励：在船位中心时给予额外奖励"""
    center_threshold = 0.2  # 中心区域阈值
    center_mask = position_error < center_threshold
    
    center_reward = torch.zeros_like(position_error)
    if torch.any(center_mask):
        # 在中心区域时，位置误差越小奖励越高
        center_reward[center_mask] = self.center_keeping_reward_scale * (
            1.0 - position_error[center_mask] / center_threshold
        )
    
    return center_reward
```

**设计思路**:
- 在船位中心区域给予额外奖励
- 位置误差越小，中心保持奖励越高
- 确保船体精确定位在中心

### **3. 智能碰撞检测**

```python
def _compute_smart_collision_penalty(self, usv_position, berth_corners, position_error):
    """智能碰撞检测：严格区分红线和绿线"""
    # 计算USV到船位中心的距离
    berth_center = (berth_corners[:, 0] + berth_corners[:, 2]) / 2
    center_distance = torch.linalg.norm(usv_position - berth_center, dim=-1)
    
    # 船位尺寸
    half_width = 1.25
    half_length = 1.75
    
    # 检查是否在船位范围内
    in_berth_x = torch.abs(usv_position[:, 0] - berth_center[:, 0]) < half_width
    in_berth_y = torch.abs(usv_position[:, 1] - berth_center[:, 1]) < half_length
    in_berth = in_berth_x & in_berth_y
    
    # 检查是否接近边界
    near_boundary = center_distance < (half_width + half_length) * 0.85
    
    # 应用碰撞惩罚
    collision_penalty = torch.zeros_like(position_error)
    
    # 绿线（入口）碰撞：轻微惩罚，允许接触
    entry_collision = in_berth & near_boundary
    collision_penalty[entry_collision] = self.entry_collision_penalty
    
    # 红线（墙壁）碰撞：极重惩罚，绝对禁止
    wall_collision = ~in_berth & near_boundary
    collision_penalty[wall_collision] = self.wall_collision_penalty
    
    return collision_penalty
```

**设计思路**:
- **红线（墙壁）**: 超出船位范围，极重惩罚(-500.0)
- **绿线（入口）**: 在船位内但接近边界，轻微惩罚(-5.0)
- 使用船位尺寸和中心距离进行判断

### **4. 时间效率奖励**

```python
def _compute_time_efficiency_reward(self, position_error, heading_error, episode_step):
    """时间效率奖励：鼓励快速完成泊船任务"""
    if episode_step is None:
        return torch.zeros_like(position_error)
    
    # 计算完成度
    position_completion = torch.clamp(1.0 - position_error / 3.0, 0.0, 1.0)
    heading_completion = torch.clamp(1.0 - heading_error / 0.5, 0.0, 1.0)
    overall_completion = (position_completion + heading_completion) / 2.0
    
    # 时间效率奖励：完成度越高，时间奖励越高
    time_reward = self.time_efficiency_scale * overall_completion
    
    # 如果已经成功，给予额外的时间奖励
    success = (position_error < self.position_tolerance) & (heading_error < self.heading_tolerance)
    if torch.any(success):
        time_reward[success] += self.max_time_bonus
    
    return time_reward
```

**设计思路**:
- 根据位置和航向完成度给予时间奖励
- 成功完成时给予额外时间奖励
- 鼓励快速完成泊船任务

### **5. 平衡的速度控制**

```python
def _compute_balanced_velocity_reward(self, current_state, position_error):
    """平衡的速度奖励：鼓励快速运动，但保持稳定性"""
    velocity_penalty = torch.zeros_like(position_error)
    velocity_encouragement = torch.zeros_like(position_error)
    
    if 'linear_velocity' in current_state:
        velocity_magnitude = torch.linalg.norm(current_state['linear_velocity'], dim=-1)
        
        # 速度惩罚：只惩罚过高速度
        high_speed_mask = velocity_magnitude > 3.0  # 允许更高的速度
        velocity_penalty[high_speed_mask] = -self.velocity_penalty_scale * (
            velocity_magnitude[high_speed_mask] - 3.0
        )
        
        # 速度鼓励：鼓励合理速度，特别是接近目标时
        reasonable_speed_mask = (velocity_magnitude > 0.5) & (velocity_magnitude < 2.5)
        velocity_encouragement[reasonable_speed_mask] = self.velocity_encouragement_scale * (
            velocity_magnitude[reasonable_speed_mask] - 0.5
        )
        
        # 在接近目标时，鼓励更快的运动
        close_target_mask = position_error < 1.0
        if torch.any(close_target_mask):
            velocity_encouragement[close_target_mask] *= 1.5
    
    return velocity_penalty, velocity_encouragement
```

**设计思路**:
- 只惩罚超过3.0的高速度
- 鼓励0.5-2.5之间的合理速度
- 接近目标时给予1.5倍速度鼓励

## 任务执行流程

### **1. 初始化阶段**
- 设置船位中心位置
- 计算船位四个角点
- 初始化episode步骤计数器

### **2. 执行阶段**
- 计算位置和航向误差
- 应用渐进式位置奖励
- 检查碰撞并应用相应惩罚
- 计算时间效率奖励
- 更新episode步骤

### **3. 重置阶段**
- 重置episode步骤计数器
- 重新生成船位位置
- 重置USV位置和状态

## 预期效果

### **1. 快速泊船**
- 高权重的位置和航向奖励
- 渐进式奖励鼓励快速接近
- 时间效率奖励避免拖延

### **2. 精确定位**
- 中心保持奖励确保在中心
- 严格的成功检测标准
- 高精度的位置和航向要求

### **3. 智能碰撞处理**
- 红线绝对禁止，确保安全
- 绿线允许接触，提供灵活性
- 区分性惩罚策略

### **4. 运动质量**
- 平衡的速度和角速度控制
- 运动平滑性奖励减少漂移
- 合理运动鼓励避免过慢

## 使用建议

### **1. 训练参数调整**
- 可以适当增加环境数量
- 可以增加batch size
- 可以适当增加学习率

### **2. 监控指标**
- 观察泊船完成时间
- 检查位置和航向精度
- 验证碰撞检测效果
- 监控运动平滑性

### **3. 进一步优化**
- 根据实际效果调整奖励权重
- 可以微调碰撞检测阈值
- 可以调整时间效率参数

## 总结

这个电子泊船任务设计专门针对无人船快速泊船需求，通过：

1. **智能奖励策略**: 渐进式奖励、中心保持奖励、时间效率奖励
2. **精确碰撞处理**: 严格区分红线和绿线，不同策略处理
3. **平衡运动控制**: 鼓励快速运动，但保持稳定性
4. **时间效率优化**: 避免拖延，鼓励快速完成

应该能够训练出快速、精确、安全的电子泊船策略。
