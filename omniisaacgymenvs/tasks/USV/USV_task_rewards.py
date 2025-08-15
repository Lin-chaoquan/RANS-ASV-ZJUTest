__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
__status__ = "development"

import torch
from dataclasses import dataclass
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class CaptureXYReward:
    """ "
    Reward function and parameters for the CaptureXY task."""

    prev_position_error = None
    reward_mode: str = "exponential"
    position_scale: float = 1.0
    exponential_reward_coeff: float = 0.25
    # r_align = La1 * exp(La2 * heading_error**4)
    align_la1: float = 0.02
    align_la2: float = -10.0
    align_la3: float = -0.1

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        position_error: torch.Tensor,
        heading_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the function used to compute the reward for the CaptureXY task."""
        if self.prev_position_error is None:
            self.prev_position_error = position_error

        if self.reward_mode.lower() == "linear":
            distance_reward = self.position_scale * (
                self.prev_position_error - position_error
            )
        elif self.reward_mode.lower() == "square":
            distance_reward = self.position_scale * (
                self.prev_position_error.pow(2) - position_error.pow(2)
            )
        elif self.reward_mode.lower() == "exponential":
            distance_reward = self.position_scale * (
                torch.exp(-position_error / self.exponential_reward_coeff)
                - torch.exp(-self.prev_position_error / self.exponential_reward_coeff)
            )
        else:
            raise ValueError("Unknown reward type.")

        alignment_reward = self.align_la1 * (
            torch.exp(self.align_la2 * heading_error.pow(4))
            + torch.exp(self.align_la3 * heading_error.pow(2))
        )

        # Update previous position error
        self.prev_position_error = position_error

        return distance_reward, alignment_reward


@dataclass
class GoToXYReward:
    """ "
    Reward function and parameters for the GoToXY task."""

    prev_position_error = None
    reward_mode: str = "exponential"
    position_scale: float = 1.0
    exponential_reward_coeff: float = 0.25
    # r_align = La1 * exp(La2 * heading_error**4)
    align_la1: float = 0.02
    align_la2: float = -10.0
    align_la3: float = -0.1

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        position_error: torch.Tensor,
        heading_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the function used to compute the reward for the GoToXY task."""
        if self.prev_position_error is None:
            self.prev_position_error = position_error

        if self.reward_mode.lower() == "linear":
            distance_reward = self.position_scale * (
                self.prev_position_error - position_error
            )
        elif self.reward_mode.lower() == "square":
            distance_reward = self.position_scale * (
                self.prev_position_error.pow(2) - position_error.pow(2)
            )
        elif self.reward_mode.lower() == "exponential":
            distance_reward = self.position_scale * (
                torch.exp(-position_error / self.exponential_reward_coeff)
                - torch.exp(-self.prev_position_error / self.exponential_reward_coeff)
            )
        else:
            raise ValueError("Unknown reward type.")

        alignment_reward = self.align_la1 * (
            torch.exp(self.align_la2 * heading_error.pow(4))
            + torch.exp(self.align_la3 * heading_error.pow(2))
        )

        # Update previous position error
        self.prev_position_error = position_error

        return distance_reward, alignment_reward


@dataclass
class GoToPoseReward:
    """
    Reward function and parameters for the GoToPose task."""

    position_reward_mode: str = "exponential"
    heading_reward_mode: str = "exponential"
    position_exponential_reward_coeff: float = 0.25
    heading_exponential_reward_coeff: float = 0.25
    position_scale: float = 1.0
    heading_scale: float = 5.0
    sig_gain: float = 3.0

    def sigmoid(self, x, gain=3.0, offset=2):
        """Sigmoid function for dynamic weighting."""
        return 1 / (1 + torch.exp(-gain * (x - offset)))

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.position_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."
        assert self.heading_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state,
        actions: torch.Tensor,
        position_error: torch.Tensor,
        heading_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the GoToPose task.
        d + k^d * h
        k^d is weighting term, where k is 0<k<1
        """
        # Adjust heading reward based on distance to goal
        heading_weight_factor = 1.0 - self.sigmoid(position_error, self.sig_gain)

        if self.position_reward_mode.lower() == "linear":
            position_reward = self.position_scale * (1.0 / (1.0 + position_error))
        elif self.position_reward_mode.lower() == "square":
            position_reward = self.position_scale * (
                1.0 / (1.0 + position_error * position_error)
            )
        elif self.position_reward_mode.lower() == "exponential":
            position_reward = self.position_scale * torch.exp(
                -position_error / self.position_exponential_reward_coeff
            )
        else:
            raise ValueError("Unknown reward type.")

        if self.heading_reward_mode.lower() == "linear":
            heading_reward = (
                heading_weight_factor
                * self.heading_scale
                * (1.0 / (1.0 + heading_error))
            )
        elif self.heading_reward_mode.lower() == "square":
            heading_reward = (
                heading_weight_factor
                * self.heading_scale
                * (1.0 / (1.0 + heading_error * heading_error))
            )
        elif self.heading_reward_mode.lower() == "exponential":
            heading_reward = (
                heading_weight_factor
                * self.heading_scale
                * torch.exp(-heading_error / self.heading_exponential_reward_coeff)
            )
        else:
            raise ValueError("Unknown reward type.")
        return position_reward, heading_reward


@dataclass
class KeepXYReward:
    """ "
    Reward function and parameters for the KeepXY task."""

    reward_mode: str = "linear"
    exponential_reward_coeff: float = 0.25

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        position_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the function used to compute the reward for the KeepXY task."""

        if self.reward_mode.lower() == "linear":
            position_reward = 1.0 / (1.0 + position_error)
        elif self.reward_mode.lower() == "square":
            position_reward = 1.0 / (1.0 + position_error * position_error)
        elif self.reward_mode.lower() == "exponential":
            position_reward = torch.exp(-position_error / self.exponential_reward_coeff)
        else:
            raise ValueError("Unknown reward type.")
        return position_reward


@dataclass
class TrackXYVelocityReward:
    """
    Reward function and parameters for the TrackXYVelocity task."""

    reward_mode: str = "exponential"
    exponential_reward_coeff: float = 0.25

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        velocity_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the TrackXYVelocity task."""

        if self.reward_mode.lower() == "linear":
            velocity_reward = 1.0 / (1.0 + velocity_error)
        elif self.reward_mode.lower() == "square":
            velocity_reward = 1.0 / (1.0 + velocity_error * velocity_error)
        elif self.reward_mode.lower() == "exponential":
            velocity_reward = torch.exp(-velocity_error / self.exponential_reward_coeff)
        else:
            raise ValueError("Unknown reward type.")
        return velocity_reward


@dataclass
class DynamicPositionReward:
    """ "
    Reward function and parameters for the DynamicPosition task with electronic anchor effect."""

    prev_position_error = None
    reward_mode: str = "exponential"
    position_scale: float = 1.0
    exponential_reward_coeff: float = 0.25
    # r_align = La1 * exp(La2 * heading_error**4)
    align_la1: float = 0.02
    align_la2: float = -10.0
    align_la3: float = -0.1
    # Electronic anchor parameters
    anchor_scale: float = 1.0  # 电子锚奖励权重
    anchor_tolerance: float = 0.1  # 电子锚容差
    velocity_penalty_scale: float = 0.1  # 速度惩罚权重
    # Yaw oscillation control parameters
    angular_velocity_penalty_scale: float = 0.3  # 角速度惩罚权重
    heading_tolerance: float = 0.1  # 航向容差 (rad)
    adaptive_alignment: bool = True  # 是否使用自适应对齐奖励

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        position_error: torch.Tensor,
        heading_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the function used to compute the reward for the DynamicPosition task with electronic anchor effect."""
        if self.prev_position_error is None:
            self.prev_position_error = position_error

        # 超稳定的距离奖励 - 确保总是正奖励
        if self.reward_mode.lower() == "linear":
            # 基础距离奖励，确保不会为负
            distance_improvement = self.prev_position_error - position_error
            base_distance_reward = self.position_scale * torch.clamp(distance_improvement, min=0.0)
            # 额外的位置奖励，基于当前位置误差
            position_reward = self.position_scale * torch.exp(-position_error / 0.5)  # 指数衰减
            distance_reward = base_distance_reward + position_reward
        elif self.reward_mode.lower() == "square":
            distance_reward = self.position_scale * (self.prev_position_error.pow(2) - position_error.pow(2))
        elif self.reward_mode.lower() == "exponential":
            distance_reward = self.position_scale * (
                torch.exp(-position_error / self.exponential_reward_coeff)
                - torch.exp(-self.prev_position_error / self.exponential_reward_coeff)
            )
        else:
            raise ValueError("Unknown reward type.")

        # 超强电子锚奖励 - 在目标附近提供极强的稳定力
        anchor_reward = torch.where(
            position_error < self.anchor_tolerance,
            self.anchor_scale * 5.0 * torch.exp(-position_error / (self.anchor_tolerance * 0.3)),  # 强指数衰减
            torch.zeros_like(position_error)
        )

        # 超强速度惩罚 - 强烈抑制任何运动
        velocity_penalty = torch.zeros_like(position_error)
        if 'linear_velocity' in current_state:
            velocity_magnitude = torch.norm(current_state['linear_velocity'], dim=-1)
            # 全局速度惩罚，但在接近目标时更强
            global_penalty = -self.velocity_penalty_scale * 0.5 * velocity_magnitude
            target_penalty = torch.where(
                position_error < self.anchor_tolerance,
                -self.velocity_penalty_scale * 2.0 * velocity_magnitude,  # 2倍惩罚
                torch.zeros_like(position_error)
            )
            velocity_penalty = global_penalty + target_penalty

        # 超强角速度惩罚 - 强烈抑制旋转
        angular_velocity_penalty = torch.zeros_like(position_error)
        if 'angular_velocity' in current_state:
            yaw_velocity = torch.abs(current_state['angular_velocity'])
            # 全局角速度惩罚，但在接近目标时更强
            global_penalty = -self.angular_velocity_penalty_scale * 0.5 * yaw_velocity
            target_penalty = torch.where(
                position_error < self.anchor_tolerance,
                -self.angular_velocity_penalty_scale * 2.0 * yaw_velocity,  # 2倍惩罚
                torch.zeros_like(position_error)
            )
            angular_velocity_penalty = global_penalty + target_penalty

        # 极简对齐奖励 - 只在非常接近目标时考虑
        if self.adaptive_alignment:
            # 只在非常接近目标时给予对齐奖励
            alignment_reward = torch.where(
                position_error < self.anchor_tolerance * 0.3,  # 更小的范围
                self.align_la1 * 0.1 * torch.exp(self.align_la2 * heading_error.pow(2)),  # 极小的权重
                torch.zeros_like(position_error)
            )
        else:
            alignment_reward = self.align_la1 * torch.exp(self.align_la2 * heading_error.pow(2))

        # Update previous position error
        self.prev_position_error = position_error

        return distance_reward + anchor_reward + velocity_penalty + angular_velocity_penalty, alignment_reward


@dataclass
class BerthingReward:
    """
    电子泊船专用奖励函数，专门针对无人船快速泊船任务优化。
    核心特性：
    1. 快速泊船：鼓励在有限时间内快速驶入船位
    2. 精确定位：保持在船位中心，避免漂移
    3. 智能碰撞处理：红线（墙壁）绝对禁止，绿线（入口）允许轻微接触
    4. 时间效率：奖励快速完成，惩罚拖延
    5. 运动质量：平衡速度和稳定性
    """

    # 基础奖励权重 - 针对快速泊船优化
    position_scale: float = 40.0  # 位置奖励权重 - 高权重确保精确定位
    heading_scale: float = 25.0  # 航向奖励权重 - 高权重确保航向对齐
    velocity_penalty_scale: float = 0.5  # 速度惩罚权重 - 低权重允许快速运动
    angular_velocity_penalty_scale: float = 1.0  # 角速度惩罚权重 - 低权重允许快速转向
    
    # 泊船特定奖励
    approach_reward_scale: float = 20.0  # 接近奖励权重 - 高权重鼓励快速接近
    center_keeping_reward_scale: float = 30.0  # 中心保持奖励权重 - 高权重确保在中心
    time_efficiency_scale: float = 15.0  # 时间效率奖励权重 - 鼓励快速完成
    
    # 方向引导奖励
    entry_direction_reward: float = 25.0  # 从绿线（入口）进入的奖励
    entry_angle_tolerance: float = 0.5  # 进入角度容差（弧度）- 放宽以允许更灵活的进入
    
    # 碰撞检测和惩罚 - 区分红线和绿线
    wall_collision_penalty: float = -500.0  # 红线（墙壁）碰撞惩罚 - 极重惩罚，绝对禁止
    entry_collision_penalty: float = -5.0  # 绿线（入口）碰撞惩罚 - 轻微惩罚，允许接触
    collision_radius: float = 0.5  # 碰撞检测半径
    
    # 成功检测
    success_reward: float = 300.0  # 成功奖励 - 高奖励强化成功行为
    position_tolerance: float = 0.1  # 位置容差 - 严格要求精确定位
    heading_tolerance: float = 0.05  # 航向容差 - 严格要求航向对齐
    
    # 时间控制
    time_penalty_scale: float = 0.1  # 时间惩罚权重 - 增加以强化时间效率
    max_time_bonus: float = 100.0  # 快速完成的最大时间奖励
    
    # 运动平滑性参数
    smoothness_reward_scale: float = 2.0  # 运动平滑性奖励权重
    velocity_encouragement_scale: float = 3.0  # 速度鼓励权重 - 鼓励合理速度
    
    # 渐进式奖励参数
    distance_thresholds: list = None  # 距离阈值列表
    reward_multipliers: list = None  # 对应奖励倍数
    
    def __post_init__(self) -> None:
        """初始化渐进式奖励参数"""
        if self.distance_thresholds is None:
            self.distance_thresholds = [2.5, 1.5, 0.8, 0.3]  # 距离阈值 - 更精细的分段
        if self.reward_multipliers is None:
            self.reward_multipliers = [0.5, 1.0, 1.8, 3.0]  # 对应奖励倍数 - 更激进的接近奖励

    def compute_reward(
        self,
        current_state: dict,
        actions: torch.Tensor,
        position_error: torch.Tensor,
        heading_error: torch.Tensor,
        berth_corners: torch.Tensor,
        usv_position: torch.Tensor,
        usv_heading: torch.Tensor,
        episode_step: int = None,
    ) -> torch.Tensor:
        """
        电子泊船专用奖励计算函数，优化快速泊船和精确定位。
        """
        
        # 1. 基础位置奖励 - 使用渐进式奖励策略
        position_reward = self._compute_progressive_position_reward(position_error)
        
        # 2. 航向奖励 - 根据距离调整权重
        heading_reward = self._compute_adaptive_heading_reward(position_error, heading_error)
        
        # 3. 中心保持奖励 - 在船位中心时给予额外奖励
        center_keeping_reward = self._compute_center_keeping_reward(position_error)
        
        # 4. 方向引导奖励 - 鼓励从绿线（入口）进入
        direction_reward = self._compute_entry_direction_reward(
            usv_position, usv_heading, berth_corners, position_error
        )
        
        # 5. 速度控制 - 平衡快速运动和稳定性
        velocity_penalty, velocity_encouragement = self._compute_balanced_velocity_reward(
            current_state, position_error
        )
        
        # 6. 角速度控制 - 平衡快速转向和稳定性
        angular_velocity_penalty, angular_velocity_encouragement = self._compute_balanced_angular_velocity_reward(
            current_state, position_error
        )
        
        # 7. 运动平滑性奖励 - 减少漂移和闪现
        smoothness_reward = self._compute_smoothness_reward(current_state, actions)
        
        # 8. 智能碰撞惩罚 - 区分红线和绿线
        collision_penalty = self._compute_smart_collision_penalty(
            usv_position, berth_corners, position_error
        )
        
        # 9. 成功奖励 - 精确定位奖励
        success_reward = self._compute_success_reward(position_error, heading_error)
        
        # 10. 时间效率奖励 - 鼓励快速完成
        time_efficiency_reward = self._compute_time_efficiency_reward(
            position_error, heading_error, episode_step
        )
        
        # 11. 组合所有奖励
        total_reward = (position_reward + heading_reward + center_keeping_reward + 
                       direction_reward + velocity_penalty + velocity_encouragement +
                       angular_velocity_penalty + angular_velocity_encouragement +
                       smoothness_reward + collision_penalty + success_reward + 
                       time_efficiency_reward)
        
        return total_reward
    
    def _compute_progressive_position_reward(self, position_error):
        """渐进式位置奖励：距离越近奖励越高，鼓励快速接近"""
        base_reward = self.position_scale * torch.exp(-position_error * 0.8)
        
        # 根据距离应用倍数，更激进的接近奖励
        reward_multiplier = torch.ones_like(position_error)
        for threshold, multiplier in zip(self.distance_thresholds, self.reward_multipliers):
            mask = position_error < threshold
            reward_multiplier[mask] = multiplier
        
        return base_reward * reward_multiplier
    
    def _compute_adaptive_heading_reward(self, position_error, heading_error):
        """自适应航向奖励：距离越近航向要求越严格"""
        # 基础航向奖励
        base_heading_reward = self.heading_scale * torch.exp(-heading_error * 2.5)
        
        # 根据距离调整权重，接近目标时更严格要求航向
        distance_factor = torch.exp(-position_error / 1.5)
        adaptive_heading_reward = base_heading_reward * distance_factor
        
        return adaptive_heading_reward
    
    def _compute_center_keeping_reward(self, position_error):
        """中心保持奖励：在船位中心时给予额外奖励"""
        # 当位置误差很小时，给予额外的中心保持奖励
        center_threshold = 0.2  # 中心区域阈值
        center_mask = position_error < center_threshold
        
        center_reward = torch.zeros_like(position_error)
        if torch.any(center_mask):
            # 在中心区域时，位置误差越小奖励越高
            center_reward[center_mask] = self.center_keeping_reward_scale * (
                1.0 - position_error[center_mask] / center_threshold
            )
        
        return center_reward
    
    def _compute_entry_direction_reward(self, usv_position, usv_heading, berth_corners, position_error):
        """计算进入方向奖励，鼓励从绿线（入口）进入"""
        # 只在距离较远时考虑方向引导
        distance_mask = position_error > 1.5
        
        if not torch.any(distance_mask):
            return torch.zeros_like(position_error)
        
        # 计算船位入口方向（绿线）
        opening_center = (berth_corners[:, 2] + berth_corners[:, 3]) / 2
        
        # 计算USV到入口的方向向量
        to_opening = opening_center - usv_position
        to_opening_angle = torch.atan2(to_opening[:, 1], to_opening[:, 0])
        
        # 计算航向误差
        heading_diff = torch.abs(torch.fmod(usv_heading - to_opening_angle + math.pi, 2 * math.pi) - math.pi)
        
        # 应用奖励
        direction_reward = torch.zeros_like(position_error)
        angle_mask = heading_diff < self.entry_angle_tolerance
        combined_mask = distance_mask & angle_mask
        
        if torch.any(combined_mask):
            direction_reward[combined_mask] = (
                self.entry_direction_reward * 
                torch.exp(-position_error[combined_mask] / 2.0)
            )
        
        return direction_reward
    
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
    
    def _compute_balanced_angular_velocity_reward(self, current_state, position_error):
        """平衡的角速度奖励：鼓励快速转向，但保持稳定性"""
        angular_velocity_penalty = torch.zeros_like(position_error)
        angular_velocity_encouragement = torch.zeros_like(position_error)
        
        if 'angular_velocity' in current_state:
            yaw_velocity = torch.abs(current_state['angular_velocity'])
            
            # 角速度惩罚：只惩罚过高角速度
            high_angular_speed_mask = yaw_velocity > 1.5  # 允许更高的角速度
            angular_velocity_penalty[high_angular_speed_mask] = -self.angular_velocity_penalty_scale * (
                yaw_velocity[high_angular_speed_mask] - 1.5
            )
            
            # 角速度鼓励：鼓励合理角速度
            reasonable_angular_speed_mask = (yaw_velocity > 0.2) & (yaw_velocity < 1.2)
            angular_velocity_encouragement[reasonable_angular_speed_mask] = self.angular_velocity_penalty_scale * 0.5 * (
                yaw_velocity[reasonable_angular_speed_mask] - 0.2
            )
        
        return angular_velocity_penalty, angular_velocity_encouragement
    
    def _compute_smoothness_reward(self, current_state, actions):
        """运动平滑性奖励：减少漂移和闪现"""
        smoothness_reward = torch.zeros_like(actions[:, 0])
        
        # 检查是否有前一步的状态
        if hasattr(self, 'prev_linear_velocity') and hasattr(self, 'prev_angular_velocity'):
            if 'linear_velocity' in current_state:
                # 线性速度变化惩罚
                velocity_change = torch.linalg.norm(
                    current_state['linear_velocity'] - self.prev_linear_velocity, dim=-1
                )
                smoothness_reward -= self.smoothness_reward_scale * velocity_change
            
            if 'angular_velocity' in current_state:
                # 角速度变化惩罚
                angular_velocity_change = torch.abs(
                    current_state['angular_velocity'] - self.prev_angular_velocity
                )
                smoothness_reward -= self.smoothness_reward_scale * angular_velocity_change
        
        # 更新前一步状态
        if 'linear_velocity' in current_state:
            self.prev_linear_velocity = current_state['linear_velocity'].clone()
        if 'angular_velocity' in current_state:
            self.prev_angular_velocity = current_state['angular_velocity'].clone()
        
        return smoothness_reward
    
    def _compute_smart_collision_penalty(self, usv_position, berth_corners, position_error):
        """
        智能碰撞检测：严格区分红线和绿线
        红线（墙壁）：绝对禁止，极重惩罚
        绿线（入口）：允许轻微接触，轻微惩罚
        """
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
    
    def _compute_success_reward(self, position_error, heading_error):
        """成功奖励：精确定位奖励"""
        success = (position_error < self.position_tolerance) & (heading_error < self.heading_tolerance)
        return torch.where(success, self.success_reward, torch.zeros_like(position_error))
    
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


@dataclass
class TrackXYOVelocityReward:
    """
    Reward function and parameters for the TrackXYOVelocity task."""

    linear_reward_mode: str = "exponential"
    angular_reward_mode: str = "exponential"
    linear_exponential_reward_coeff: float = 0.25
    angular_exponential_reward_coeff: float = 0.25
    linear_scale: float = 1.0
    angular_scale: float = 1.0

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.linear_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."
        assert self.angular_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state,
        actions: torch.Tensor,
        linear_velocity_error: torch.Tensor,
        angular_velocity_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the TrackXYOVelocity task.
        """

        if self.linear_reward_mode.lower() == "linear":
            linear_reward = 1.0 / (1.0 + linear_velocity_error) * self.linear_scale
        elif self.linear_reward_mode.lower() == "square":
            linear_reward = 1.0 / (1.0 + linear_velocity_error**2) * self.linear_scale
        elif self.linear_reward_mode.lower() == "exponential":
            linear_reward = (
                torch.exp(-linear_velocity_error / self.linear_exponential_reward_coeff)
                * self.linear_scale
            )
        else:
            raise ValueError("Unknown reward type.")

        if self.angular_reward_mode.lower() == "linear":
            angular_reward = 1.0 / (1.0 + angular_velocity_error) * self.angular_scale
        elif self.angular_reward_mode.lower() == "square":
            angular_reward = (
                1.0 / (1.0 + angular_velocity_error**2) * self.angular_scale
            )
        elif self.angular_reward_mode.lower() == "exponential":
            angular_reward = (
                torch.exp(
                    -angular_velocity_error / self.angular_exponential_reward_coeff
                )
                * self.angular_scale
            )
        else:
            raise ValueError("Unknown reward type.")
        return linear_reward, angular_reward


@dataclass
class Penalties:
    """
    Metaclass to compute penalties for the tasks."""

    prev_state = None
    prev_actions = None

    penalize_linear_velocities: bool = False
    penalize_linear_velocities_fn: str = (
        "lambda x,step : -torch.norm(x, dim=-1)*c1 + c2"
    )
    penalize_linear_velocities_c1: float = 0.01
    penalize_linear_velocities_c2: float = 0.0
    penalize_angular_velocities: bool = False
    penalize_angular_velocities_fn: str = "lambda x,step : -torch.abs(x)*c1 + c2"
    penalize_angular_velocities_c1: float = 0.01
    penalize_angular_velocities_c2: float = 0.0
    penalize_angular_velocities_variation: bool = False
    penalize_angular_velocities_variation_fn: str = (
        "lambda x,step: torch.exp(c1 * torch.abs(x)) - 1.0"
    )
    penalize_angular_velocities_variation_c1: float = -0.033
    penalize_energy: bool = False
    penalize_energy_fn: str = "lambda x,step : -torch.sum(x**2)*c1 + c2"
    penalize_energy_c1: float = 0.01
    penalize_energy_c2: float = 0.0
    penalize_action_variation: bool = False
    penalize_action_variation_fn: str = (
        "lambda x,step: torch.exp(c1 * torch.abs(x)) - 1.0"
    )
    penalize_action_variation_c1: float = -0.033

    def __post_init__(self):
        """
        Converts the string functions into python callable functions."""
        self.penalize_linear_velocities_fn = eval(self.penalize_linear_velocities_fn)
        self.penalize_angular_velocities_fn = eval(self.penalize_angular_velocities_fn)
        self.penalize_angular_velocities_variation_fn = eval(
            self.penalize_angular_velocities_variation_fn
        )
        self.penalize_energy_fn = eval(self.penalize_energy_fn)
        self.penalize_action_variation_fn = eval(self.penalize_action_variation_fn)

    def compute_penalty(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Computes the penalties for the task."""

        # Initialize previous state and action
        if self.prev_state is None:
            self.prev_state = state
        if self.prev_actions is None:
            self.prev_actions = actions

        # Linear velocity penalty
        if self.penalize_linear_velocities:
            self.linear_vel_penalty = self.penalize_linear_velocities_fn(
                state["linear_velocity"],
                torch.tensor(step, dtype=torch.float32, device=actions.device),
            )
        else:
            self.linear_vel_penalty = torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )
        # Angular velocity penalty
        if self.penalize_angular_velocities:
            self.angular_vel_penalty = self.penalize_angular_velocities_fn(
                state["angular_velocity"],
                torch.tensor(step, dtype=torch.float32, device=actions.device),
            )
        else:
            self.angular_vel_penalty = torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )
        # Angular velocity variation penalty
        if self.penalize_angular_velocities_variation:
            self.angular_vel_variation_penalty = (
                self.penalize_angular_velocities_variation_fn(
                    state["angular_velocity"] - self.prev_state["angular_velocity"],
                    torch.tensor(step, dtype=torch.float32, device=actions.device),
                )
            )
        else:
            self.angular_vel_variation_penalty = torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )
        # Energy penalty
        if self.penalize_energy:
            self.energy_penalty = self.penalize_energy_fn(
                actions,
                torch.tensor(step, dtype=torch.float32, device=actions.device),
            )
        else:
            self.energy_penalty = torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )
        # Action variation penalty
        if self.penalize_action_variation:
            self.action_variation_penalty = self.penalize_action_variation_fn(
                torch.sum(actions, dim=-1) - torch.sum(self.prev_actions, dim=-1),
                torch.tensor(step, dtype=torch.float32, device=actions.device),
            )
        else:
            self.action_variation_penalty = torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )

        # print penalties
        # print("linear_vel_penalty: ", self.linear_vel_penalty)
        # print("angular_vel_penalty: ", self.angular_vel_penalty)
        # print("energy_penalty: ", self.energy_penalty)

        # Update previous state and action
        self.prev_state = state
        self.prev_actions = actions

        return (
            self.linear_vel_penalty
            + self.angular_vel_penalty
            + self.angular_vel_variation_penalty
            + self.energy_penalty
            + self.action_variation_penalty
        )

    def get_stats_name(self) -> list:
        """
        Returns the names of the statistics to be computed."""

        names = []
        if self.penalize_linear_velocities:
            names.append("linear_vel_penalty")
        if self.penalize_angular_velocities:
            names.append("angular_vel_penalty")
        if self.penalize_angular_velocities_variation:
            names.append("angular_vel_variation_penalty")
        if self.penalize_energy:
            names.append("energy_penalty")
        if self.penalize_action_variation:
            names.append("action_variation_penalty")
        return names

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics."""

        if self.penalize_linear_velocities:
            stats["linear_vel_penalty"] += self.linear_vel_penalty
        if self.penalize_angular_velocities:
            stats["angular_vel_penalty"] += self.angular_vel_penalty
        if self.penalize_angular_velocities_variation:
            stats["angular_vel_variation_penalty"] += self.angular_vel_variation_penalty
        if self.penalize_energy:
            stats["energy_penalty"] += self.energy_penalty
        if self.penalize_action_variation:
            stats["action_variation_penalty"] += self.action_variation_penalty
        return stats
