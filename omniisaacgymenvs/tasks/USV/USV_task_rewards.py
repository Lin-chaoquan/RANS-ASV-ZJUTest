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
    Reward function and parameters for the Berthing task.
    优化版本：向量化计算，减少循环，提高运行速度。
    """

    # 基础奖励权重
    position_scale: float = 15.0  # 位置奖励权重
    heading_scale: float = 8.0  # 航向奖励权重
    velocity_penalty_scale: float = 3.0  # 速度惩罚权重
    angular_velocity_penalty_scale: float = 4.0  # 角速度惩罚权重
    
    # 泊船特定奖励
    approach_reward_scale: float = 5.0  # 接近奖励权重
    alignment_reward_scale: float = 3.0  # 对齐奖励权重
    stability_reward_scale: float = 2.0  # 稳定性奖励权重
    
    # 碰撞检测
    collision_penalty: float = -50.0  # 碰撞惩罚
    wall_distance_threshold: float = 0.8  # 墙距离阈值
    
    # 成功检测
    success_reward: float = 100.0  # 成功奖励
    position_tolerance: float = 0.2  # 位置容差
    heading_tolerance: float = 0.1  # 航向容差
    
    # 时间控制
    time_penalty_scale: float = 0.01  # 时间惩罚权重

    def __post_init__(self) -> None:
        """Checks that the reward parameters are valid."""
        pass

    def compute_reward(
        self,
        current_state: dict,
        actions: torch.Tensor,
        position_error: torch.Tensor,
        heading_error: torch.Tensor,
        berth_corners: torch.Tensor,
        collision_radius: float = 0.5,
    ) -> torch.Tensor:
        """
        优化的奖励计算函数，使用向量化操作提高速度。
        """
        
        # 1. 位置奖励 - 使用更高效的指数计算
        position_reward = self.position_scale * torch.exp(-position_error * 0.5)
        
        # 2. 航向奖励 - 使用更高效的指数计算
        heading_reward = self.heading_scale * torch.exp(-heading_error * 2.0)
        
        # 3. 速度惩罚 - 向量化计算
        velocity_penalty = torch.zeros_like(position_error)
        if 'linear_velocity' in current_state:
            # 使用更高效的范数计算
            velocity_magnitude = torch.linalg.norm(current_state['linear_velocity'], dim=-1)
            velocity_penalty = -self.velocity_penalty_scale * velocity_magnitude
        
        # 4. 角速度惩罚 - 向量化计算
        angular_velocity_penalty = torch.zeros_like(position_error)
        if 'angular_velocity' in current_state:
            # 直接使用绝对值，避免不必要的计算
            yaw_velocity = torch.abs(current_state['angular_velocity'])
            angular_velocity_penalty = -self.angular_velocity_penalty_scale * yaw_velocity
        
        # 5. 碰撞惩罚 - 向量化碰撞检测
        collision_penalty = self._check_collision_with_berth_vectorized(
            current_state['position'], berth_corners, collision_radius
        )
        
        # 6. 成功奖励 - 向量化计算
        success_reward = self._check_success_vectorized(position_error, heading_error)
        
        # 7. 时间惩罚 - 预计算
        time_penalty = -self.time_penalty_scale * torch.ones_like(position_error)
        
        # 8. 组合所有奖励 - 一次性计算
        total_reward = (position_reward + heading_reward + velocity_penalty + 
                       angular_velocity_penalty + collision_penalty + success_reward + time_penalty)
        
        return total_reward
    
    def _check_collision_with_berth_vectorized(self, usv_position, berth_corners, collision_radius):
        """
        向量化碰撞检测，避免Python循环，大幅提高速度。
        berth_corners: [num_envs, 4, 2] 每个环境的船位四个角点
        usv_position: [num_envs, 2] USV位置
        """
        num_envs = usv_position.shape[0]
        
        # 预分配碰撞惩罚张量
        collision_penalty = torch.zeros(num_envs, device=usv_position.device, dtype=usv_position.dtype)
        
        # 向量化计算所有边界的距离
        # 重新排列berth_corners以便向量化计算
        # [num_envs, 4, 2] -> [num_envs, 4, 2] (保持原形状)
        
        # 计算所有边的起点和终点
        wall_starts = berth_corners  # [num_envs, 4, 2]
        wall_ends = torch.roll(berth_corners, shifts=1, dims=1)  # [num_envs, 4, 2]
        
        # 向量化计算到所有边的距离
        # 使用广播机制一次性计算所有环境的所有边
        distances = self._point_to_line_distance_vectorized(
            usv_position.unsqueeze(1).expand(-1, 4, -1),  # [num_envs, 4, 2]
            wall_starts,  # [num_envs, 4, 2]
            wall_ends     # [num_envs, 4, 2]
        )  # [num_envs, 4]
        
        # 找到每个环境的最小距离（最接近的边）
        min_distances = torch.min(distances, dim=1)[0]  # [num_envs]
        
        # 向量化应用碰撞惩罚
        collision_mask = min_distances < collision_radius
        collision_penalty[collision_mask] = self.collision_penalty
        
        return collision_penalty
    
    def _point_to_line_distance_vectorized(self, points, line_starts, line_ends):
        """
        向量化计算点到线段的距离。
        points: [num_envs, num_lines, 2]
        line_starts: [num_envs, num_lines, 2]
        line_ends: [num_envs, num_lines, 2]
        返回: [num_envs, num_lines]
        """
        # 计算线段向量
        line_vecs = line_ends - line_starts  # [num_envs, num_lines, 2]
        
        # 计算点到起点的向量
        point_vecs = points - line_starts  # [num_envs, num_lines, 2]
        
        # 计算线段长度（避免除零）
        line_lengths = torch.linalg.norm(line_vecs, dim=-1)  # [num_envs, num_lines]
        line_lengths = torch.clamp(line_lengths, min=1e-6)
        
        # 计算投影参数 t
        # 使用向量点积计算投影
        dot_products = torch.sum(point_vecs * line_vecs, dim=-1)  # [num_envs, num_lines]
        t_values = dot_products / (line_lengths ** 2)  # [num_envs, num_lines]
        t_values = torch.clamp(t_values, 0, 1)
        
        # 计算最近点
        closest_points = line_starts + t_values.unsqueeze(-1) * line_vecs  # [num_envs, num_lines, 2]
        
        # 计算距离
        distances = torch.linalg.norm(points - closest_points, dim=-1)  # [num_envs, num_lines]
        
        return distances
    
    def _check_success_vectorized(self, position_error, heading_error):
        """
        向量化成功检测，避免循环。
        """
        position_success = position_error < self.position_tolerance
        heading_success = heading_error < self.heading_tolerance
        success = position_success & heading_success
        
        return torch.where(success, self.success_reward, torch.zeros_like(position_error))
    
    # 保留旧方法作为备用（如果需要的话）
    def _check_collision_with_berth(self, usv_position, berth_corners, collision_radius):
        """
        旧的循环版本碰撞检测（保留作为备用）。
        """
        num_envs = usv_position.shape[0]
        collision_penalty = torch.zeros(num_envs, device=usv_position.device, dtype=usv_position.dtype)
        
        for env_id in range(num_envs):
            for i in range(4):
                wall_start = berth_corners[env_id, i]
                wall_end = berth_corners[env_id, (i + 1) % 4]
                
                wall_distance = self._point_to_line_distance(
                    usv_position[env_id:env_id+1], 
                    wall_start.unsqueeze(0), 
                    wall_end.unsqueeze(0)
                )
                
                if wall_distance < collision_radius:
                    collision_penalty[env_id] = self.collision_penalty
                    break
        
        return collision_penalty
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的距离（旧版本，保留作为备用）"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length = torch.norm(line_vec, dim=-1)
        line_length = torch.clamp(line_length, min=1e-6)
        
        t = torch.sum(point_vec * line_vec, dim=-1) / (line_length ** 2)
        t = torch.clamp(t, 0, 1)
        
        closest_point = line_start + t.unsqueeze(-1) * line_vec
        
        return torch.norm(point - closest_point, dim=-1)
    
    def _check_success(self, position_error, heading_error):
        """检查是否成功泊船（旧版本，保留作为备用）"""
        position_success = position_error < self.position_tolerance
        heading_success = heading_error < self.heading_tolerance
        success = position_success & heading_success
        
        return torch.where(success, self.success_reward, torch.zeros_like(position_error))

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
