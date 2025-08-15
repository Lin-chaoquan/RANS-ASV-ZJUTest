# 电子泊船任务碰撞检测逻辑详解

## 概述

电子泊船任务中的碰撞检测是一个关键组件，它严格区分了**红线（墙壁）**和**绿线（入口）**，采用不同的惩罚策略来确保安全性和灵活性。

## 碰撞检测架构

### **1. 双层碰撞检测系统**

电子泊船任务采用了双层碰撞检测系统：

1. **任务层碰撞检测** (`_check_collision_simple`): 用于任务状态管理
2. **奖励层碰撞检测** (`_compute_smart_collision_penalty`): 用于奖励计算和惩罚

### **2. 碰撞检测频率优化**

```python
# 每10步检测一次碰撞，避免过度计算
if current_step - self._last_collision_check_step > 10:
    self.collision_detected = self._check_collision_simple(...)
    self._last_collision_check_step = current_step
```

## 详细碰撞检测逻辑

### **1. 任务层碰撞检测 (`_check_collision_simple`)**

#### **目的**: 快速判断是否发生碰撞，用于任务状态管理

#### **检测逻辑**:
```python
def _check_collision_simple(self, env_ids: torch.Tensor) -> torch.Tensor:
    """简化的碰撞检测方法，提高性能"""
    
    # 1. 获取船位中心和USV位置
    berth_centers = self._berth_centers[env_ids]  # [num_resets, 2]
    usv_positions = self._position_error[env_ids, :2]  # [num_resets, 2]
    
    # 2. 计算USV到船位中心的距离
    distances = torch.linalg.norm(usv_positions - berth_centers, dim=-1)
    
    # 3. 计算船位对角线长度
    berth_diagonal = torch.sqrt(
        self._task_parameters.berth_width ** 2 + 
        self._task_parameters.berth_length ** 2
    )
    
    # 4. 设置碰撞阈值（船位对角线的40%）
    collision_threshold = berth_diagonal * 0.4
    
    # 5. 判断是否发生碰撞
    collisions = distances < collision_threshold
    
    return collisions
```

#### **特点**:
- **快速**: 只计算到中心的距离，避免复杂几何计算
- **简化**: 使用对角线长度作为碰撞判断依据
- **高效**: 适合频繁调用的任务状态管理

### **2. 奖励层碰撞检测 (`_compute_smart_collision_penalty`)**

#### **目的**: 精确区分碰撞类型，应用相应的惩罚策略

#### **检测逻辑**:
```python
def _compute_smart_collision_penalty(self, usv_position, berth_corners, position_error):
    """智能碰撞检测：严格区分红线和绿线"""
    
    # 1. 计算船位中心
    berth_center = (berth_corners[:, 0] + berth_corners[:, 2]) / 2
    
    # 2. 计算USV到船位中心的距离
    center_distance = torch.linalg.norm(usv_position - berth_center, dim=-1)
    
    # 3. 定义船位尺寸
    half_width = 1.25   # 船位宽度的一半
    half_length = 1.75  # 船位长度的一半
    
    # 4. 检查USV是否在船位范围内
    in_berth_x = torch.abs(usv_position[:, 0] - berth_center[:, 0]) < half_width
    in_berth_y = torch.abs(usv_position[:, 1] - berth_center[:, 1]) < half_length
    in_berth = in_berth_x & in_berth_y
    
    # 5. 检查是否接近边界
    near_boundary = center_distance < (half_width + half_length) * 0.85
    
    # 6. 应用不同的碰撞惩罚
    collision_penalty = torch.zeros_like(position_error)
    
    # 绿线（入口）碰撞：轻微惩罚，允许接触
    entry_collision = in_berth & near_boundary
    collision_penalty[entry_collision] = self.entry_collision_penalty  # -5.0
    
    # 红线（墙壁）碰撞：极重惩罚，绝对禁止
    wall_collision = ~in_berth & near_boundary
    collision_penalty[wall_collision] = self.wall_collision_penalty   # -500.0
    
    return collision_penalty
```

## 碰撞类型分类

### **1. 绿线（入口）碰撞**

#### **定义**: USV在船位范围内，但接近边界

#### **判断条件**:
```python
# 1. 在船位范围内
in_berth = (|x - center_x| < half_width) & (|y - center_y| < half_length)

# 2. 接近边界
near_boundary = distance_to_center < (half_width + half_length) * 0.85

# 3. 绿线碰撞
entry_collision = in_berth & near_boundary
```

#### **惩罚策略**:
- **惩罚值**: -5.0（轻微惩罚）
- **策略**: 允许轻微接触，提供灵活性
- **目的**: 让USV能够利用入口区域进行微调

### **2. 红线（墙壁）碰撞**

#### **定义**: USV超出船位范围，接近边界

#### **判断条件**:
```python
# 1. 超出船位范围
~in_berth = (|x - center_x| >= half_width) | (|y - center_y| >= half_length)

# 2. 接近边界
near_boundary = distance_to_center < (half_width + half_length) * 0.85

# 3. 红线碰撞
wall_collision = ~in_berth & near_boundary
```

#### **惩罚策略**:
- **惩罚值**: -500.0（极重惩罚）
- **策略**: 绝对禁止，确保安全
- **目的**: 防止USV撞墙，保护船体和环境

## 几何参数说明

### **1. 船位尺寸**

| 参数 | 数值 | 说明 |
|------|------|------|
| `berth_width` | 2.5 | 船位宽度（米） |
| `berth_length` | 3.5 | 船位长度（米） |
| `half_width` | 1.25 | 船位宽度的一半 |
| `half_length` | 1.75 | 船位长度的一半 |

### **2. 碰撞检测阈值**

| 检测类型 | 阈值计算 | 数值 | 说明 |
|----------|----------|------|------|
| **任务层碰撞** | `berth_diagonal * 0.4` | ≈1.0 | 快速碰撞检测 |
| **奖励层边界** | `(half_width + half_length) * 0.85` | ≈2.55 | 精确边界检测 |

### **3. 船位角点定义**

```python
# 船位四个角点的相对位置
berth_corners = [
    [center_x - half_width, center_y - half_length],  # 左下角
    [center_x + half_width, center_y - half_length],  # 右下角
    [center_x + half_width, center_y + half_length],  # 右上角（入口）
    [center_x - half_width, center_y + half_length]   # 左上角（入口）
]
```

## 碰撞检测流程

### **1. 初始化阶段**

```python
# 1. 生成船位中心位置
self._berth_centers[env_ids] = random_offset

# 2. 计算船位角点
self._calculate_berth_corners(env_ids)

# 3. 初始化碰撞状态
self.collision_detected = torch.zeros(self._num_envs, dtype=torch.bool)
```

### **2. 执行阶段**

```python
# 1. 每10步检测一次碰撞（任务层）
if current_step - self._last_collision_check_step > 10:
    self.collision_detected = self._check_collision_simple(...)

# 2. 每步计算奖励时检测碰撞（奖励层）
collision_penalty = self._compute_smart_collision_penalty(...)
```

### **3. 重置阶段**

```python
# 1. 重置碰撞状态
self.collision_detected[env_ids] = False

# 2. 重新生成船位位置
# 在get_goals中完成
```

## 性能优化策略

### **1. 检测频率优化**

- **任务层**: 每10步检测一次，减少计算开销
- **奖励层**: 每步检测，确保奖励准确性

### **2. 计算复杂度优化**

- **任务层**: O(n)复杂度，n为环境数量
- **奖励层**: O(n)复杂度，避免复杂的点到线段距离计算

### **3. 内存优化**

- 重用张量，避免频繁的内存分配
- 使用向量化操作，提高GPU利用率

## 碰撞检测的优势

### **1. 安全性**

- **红线绝对禁止**: 防止USV撞墙
- **实时检测**: 及时发现潜在危险

### **2. 灵活性**

- **绿线允许接触**: 提供微调空间
- **渐进式惩罚**: 根据碰撞严重程度调整惩罚

### **3. 性能**

- **分层检测**: 任务层快速，奖励层精确
- **频率优化**: 避免过度计算
- **向量化**: 充分利用GPU并行计算能力

## 使用建议

### **1. 参数调整**

- **碰撞阈值**: 根据实际船位尺寸调整
- **惩罚权重**: 根据训练效果调整红线和绿线的惩罚比例

### **2. 监控指标**

- **碰撞频率**: 观察红线碰撞是否减少
- **入口利用**: 观察绿线接触是否合理
- **训练稳定性**: 确保惩罚不会导致训练崩溃

### **3. 进一步优化**

- **自适应阈值**: 根据训练进度动态调整碰撞阈值
- **历史记录**: 记录碰撞历史，分析碰撞模式
- **预测检测**: 预测潜在碰撞，提前避免

## 总结

电子泊船任务的碰撞检测系统通过：

1. **双层架构**: 任务层快速检测 + 奖励层精确惩罚
2. **智能分类**: 严格区分红线和绿线，不同策略处理
3. **性能优化**: 频率控制 + 向量化计算 + 内存优化
4. **安全可靠**: 红线绝对禁止，绿线允许接触

实现了安全、灵活、高效的碰撞检测，为无人船电子泊船提供了可靠的安全保障。
