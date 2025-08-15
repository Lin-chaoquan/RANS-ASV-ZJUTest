from omniisaacgymenvs.tasks.USV.USV_core import (
    Core,
    parse_data_dict,
)
from omniisaacgymenvs.tasks.USV.USV_task_rewards import (
    BerthingReward,
)
from omniisaacgymenvs.tasks.USV.USV_task_parameters import (
    BerthingParameters,
)
from omniisaacgymenvs.utils.pin import VisualPin, VisualRectangleLine
# 暂时注释掉，使用简单的线条绘制
# from omniisaacgymenvs.utils.pin_fixed import VisualRectangleLine

from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf
import omni.usd

import math
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class BerthingTask(Core):
    """
    Implements the Berthing task. The robot has to berth into a rectangular berth."""

    def __init__(
        self,
        task_param: BerthingParameters,
        reward_param: BerthingReward,
        num_envs: int,
        device: str,
    ) -> None:
        super(BerthingTask, self).__init__(num_envs, device)
        # Task and reward parameters
        self._task_parameters = parse_data_dict(BerthingParameters(), task_param)
        self._reward_parameters = parse_data_dict(BerthingReward(), reward_param)

        # Buffers
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        # 每个环境都有自己的船位中心
        self._berth_centers = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        # 重置任务标签
        self._task_label = torch.zeros_like(self._task_label)
        self.just_had_been_reset = torch.arange(
            0, num_envs, device=self._device, dtype=torch.long
        )

        # 初始化船位角点（将在reset时更新）
        self._berth_corners = torch.zeros(
            (self._num_envs, 4, 2), device=self._device, dtype=torch.float32
        )
        
        # 船位中心设置：参考CaptureXY的实现方式
        # 不在初始化时设置固定位置，而是在get_goals中动态生成
        print(f"=== 泊船任务初始化 ===")
        print(f"环境数量: {self._num_envs}")
        
        
        print(f"船位中心缓冲区已初始化，位置将在get_goals中设置")
        print(f"=== 泊船任务初始化完成 ===")
        # 基于船位中心计算四个角点
        self._calculate_berth_corners()

        # 初始化统计变量
        self.prev_position_dist = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self.position_dist = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self.heading_error = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self.combined_reward = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self.collision_detected = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._position_error = torch.zeros(self._num_envs, 2, device=self._device, dtype=torch.float32)

    def _calculate_berth_corners(self, env_ids: torch.Tensor = None):
        """
        优化的船位角点计算方法，使用向量化操作提高性能。
        计算矩形泊船位的四个角点坐标。
        """
        
        # 基础角点（相对于原点）
        half_width = self._task_parameters.berth_width / 2
        half_length = self._task_parameters.berth_length / 2
        
        # 预计算基础角点，避免重复计算
        base_corners = torch.tensor([
            [-half_width, -half_length],  # [0] 左下角
            [ half_width, -half_length],  # [1] 右下角
            [ half_width,  half_length],  # [2] 右上角
            [-half_width,  half_length],  # [3] 左上角
        ], device=self._device, dtype=torch.float32)
        
        if env_ids is None:
            # 为所有环境计算 - 向量化操作
            # 使用广播机制一次性计算所有环境
            # [num_envs, 1, 2] + [1, 4, 2] = [num_envs, 4, 2]
            self._berth_corners = self._berth_centers.unsqueeze(1) + base_corners.unsqueeze(0)
        else:
            # 只为指定环境计算 - 向量化操作
            # [len(env_ids), 1, 2] + [1, 4, 2] = [len(env_ids), 4, 2]
            self._berth_corners[env_ids] = self._berth_centers[env_ids].unsqueeze(1) + base_corners.unsqueeze(0)

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task."""

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float, device=self._device, requires_grad=False
        )

        if not "combined_reward" in stats.keys():
            stats["combined_reward"] = torch_zeros()
        if not "position_error" in stats.keys():
            stats["position_error"] = torch_zeros()
        if not "heading_error" in stats.keys():
            stats["heading_error"] = torch_zeros()
        if not "collision_detected" in stats.keys():
            stats["collision_detected"] = torch_zeros()
        if not "berth_success" in stats.keys():
            stats["berth_success"] = torch_zeros()
        return stats

    def get_state_observations(
        self, current_state: dict, observation_frame: str
    ) -> torch.Tensor:
        """
        优化的状态观察计算，减少重复计算，提高性能。
        """

        # 1. 计算位置误差 - 向量化操作
        self._position_error = self._berth_centers - current_state["position"][:, :2]
        
        # 2. 计算航向误差 - 向量化操作
        # 当前航向
        theta = torch.atan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        
        # 目标航向（朝向船位中心）
        beta = torch.atan2(self._position_error[:, 1], self._position_error[:, 0])
        
        # 航向误差（归一化到[-π, π]）
        alpha = torch.fmod(beta - theta + math.pi, 2 * math.pi) - math.pi
        self.heading_error = torch.abs(alpha)
        
        # 3. 更新任务数据 - 向量化操作
        self._task_data[:, 0] = torch.cos(alpha)
        self._task_data[:, 1] = torch.sin(alpha)
        self._task_data[:, 2] = torch.linalg.norm(self._position_error, dim=1)
        
        # 4. 检查碰撞 - 向量化碰撞检测
        self.collision_detected = self._check_collision(
            torch.arange(self._num_envs, device=self._device)
        )
        
        # 5. 返回观察张量
        return self.update_observation_tensor(current_state, observation_frame)

    def _check_collision(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        优化的碰撞检测方法，使用向量化操作提高性能。
        检查USV是否与船位边界发生碰撞。
        """
        
        if len(env_ids) == 0:
            return torch.tensor([], device=self._device, dtype=torch.bool)
        
        # 获取需要检查的环境的船位角点
        berth_corners = self._berth_corners[env_ids]  # [num_resets, 4, 2]
        usv_positions = self._position_error[env_ids, :2]  # [num_resets, 2]
        
        # 向量化计算所有边界的距离
        # 计算所有边的起点和终点
        wall_starts = berth_corners  # [num_resets, 4, 2]
        wall_ends = torch.roll(berth_corners, shifts=1, dims=1)  # [num_resets, 4, 2]
        
        # 向量化计算到所有边的距离
        distances = self._point_to_line_distance_vectorized(
            usv_positions.unsqueeze(1).expand(-1, 4, -1),  # [num_resets, 4, 2]
            wall_starts,  # [num_resets, 4, 2]
            wall_ends     # [num_resets, 4, 2]
        )  # [num_resets, 4]
        
        # 找到每个环境的最小距离（最接近的边）
        min_distances = torch.min(distances, dim=1)[0]  # [num_resets]
        
        # 检查是否发生碰撞（距离小于碰撞半径）
        collision_radius = getattr(self._task_parameters, 'collision_radius', 0.5)
        collisions = min_distances < collision_radius
        
        # 更新碰撞状态
        self.collision_detected[env_ids] = collisions
        
        return collisions
    
    def _point_to_line_distance_vectorized(self, points, line_starts, line_ends):
        """
        向量化计算点到线段的距离，避免循环。
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
        dot_products = torch.sum(point_vecs * line_vecs, dim=-1)  # [num_envs, num_lines]
        t_values = dot_products / (line_lengths ** 2)  # [num_envs, num_lines]
        t_values = torch.clamp(t_values, 0, 1)
        
        # 计算最近点
        closest_points = line_starts + t_values.unsqueeze(-1) * line_vecs  # [num_envs, num_lines, 2]
        
        # 计算距离
        distances = torch.linalg.norm(points - closest_points, dim=-1)  # [num_envs, num_lines]
        
        return distances

    def _generate_berth_centers(self, env_ids):
        """为指定环境生成船位中心"""
        # 此方法已废弃，船位中心生成逻辑已移至get_goals方法
        # 保持向后兼容，但不执行任何操作
        print(f"_generate_berth_centers: 此方法已废弃，船位中心在get_goals中生成")
        pass



    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        优化的奖励计算函数，减少重复计算，提高性能。
        """

        # 1. 计算位置距离 - 使用更高效的范数计算
        self.position_dist = torch.linalg.norm(self._position_error, dim=-1)
        
        # 2. 计算组合奖励 - 一次性调用
        self.combined_reward = self._reward_parameters.compute_reward(
            current_state,
            actions,
            self.position_dist,
            self.heading_error,
            self._berth_corners,
            self._task_parameters.collision_radius,
        )

        # 3. 重置时奖励为0 - 向量化操作
        if len(self.just_had_been_reset) > 0:
            self.combined_reward[self.just_had_been_reset] = 0
            self.just_had_been_reset = torch.tensor(
                [], device=self._device, dtype=torch.long
            )

        # 4. 检查成功状态 - 向量化计算
        success = self._check_success()
        goal_reward = success * self._task_parameters.goal_reward
        
        # 5. 时间惩罚 - 预计算
        time_reward = self._task_parameters.time_reward

        # 6. 返回总奖励 - 一次性计算
        return self.combined_reward + goal_reward + time_reward

    def _check_success(self):
        """
        向量化检查是否成功泊船，避免循环。
        """
        # 向量化比较操作
        position_success = self.position_dist < self._task_parameters.position_tolerance
        heading_success = self.heading_error < self._task_parameters.heading_tolerance
        
        # 向量化逻辑与操作
        success = position_success & heading_success
        
        # 转换为浮点数
        return success.float()

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics."""

        stats["combined_reward"] += self.combined_reward
        stats["position_error"] += self.position_dist
        stats["heading_error"] += self.heading_error
        stats["collision_detected"] += self.collision_detected.float()
        stats["berth_success"] += self._check_success()
        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task.
        参考CaptureXY的实现方式，船位中心在get_goals中生成。
        """

        self._goal_reached[env_ids] = 0
        self.just_had_been_reset = env_ids.clone()
        
        # 确保船位中心被正确设置（如果还没有设置的话）
        # 这是为了处理在get_goals被调用之前就调用get_state_observations或get_spawns的情况
        if torch.all(self._berth_centers[env_ids] == 0):
            print(f"reset: 环境 {env_ids} 的船位中心还未设置，生成默认值")
            # 完全按照CaptureXY：为这些环境生成船位中心
            random_range = float(getattr(self._task_parameters, "goal_random_position", 0.0))
            if random_range > 0.0:
                # 在原点附近添加随机偏移
                self._berth_centers[env_ids] = (
                    torch.rand((len(env_ids), 2), device=self._device) * 2.0 - 1.0
                ) * random_range
            else:
                # 固定在环境原点
                self._berth_centers[env_ids] = 0.0
            print(f"reset: 为环境 {env_ids} 生成船位中心: {self._berth_centers[env_ids]}")
        
        # 更新船位角点（基于当前的船位中心）
        self._calculate_berth_corners(env_ids)
        
        print(f"reset: 环境 {env_ids} 重置完成，船位中心: {self._berth_centers[env_ids]}")

    def get_goals(
        self,
        env_ids: torch.Tensor,
        targets_position: torch.Tensor,
        targets_orientation: torch.Tensor,
    ) -> list:
        """
        完全按照CaptureXY的实现方式，为每个环境生成船位中心目标。
        船位中心位置在每次重置时动态生成，确保每个环境都有自己的目标。
        """

        num_goals = len(env_ids)
        
        # 完全按照CaptureXY的实现：
        # 1. 生成目标位置（相对于环境原点的偏移）
        # 2. 使用+=操作添加到传入的targets_position
        random_range = float(getattr(self._task_parameters, "goal_random_position", 0.0))
        
        if random_range > 0.0:
            # 在原点附近添加随机偏移
            self._berth_centers[env_ids] = (
                torch.rand((num_goals, 2), device=self._device) * 2.0 - 1.0
            ) * random_range
        else:
            # 固定在环境原点
            self._berth_centers[env_ids] = 0.0
        
        # 完全按照CaptureXY：使用+=操作，将目标位置作为偏移量添加
        targets_position[env_ids, :2] += self._berth_centers[env_ids]
        
        print(f"get_goals: 为环境 {env_ids} 生成船位中心偏移: {self._berth_centers[env_ids]}")
        print(f"get_goals: 最终目标位置: {targets_position[env_ids, :2]}")
        
        return targets_position, targets_orientation

    def get_spawns(
        self,
        env_ids: torch.Tensor,
        initial_position: torch.Tensor,
        initial_orientation: torch.Tensor,
        step: int = 0,
    ) -> list:
        """
        完全按照CaptureXY的实现方式，生成船体生成位置。
        船体在距离目标位置（船位中心）的环形区域内生成。
        """

        num_resets = len(env_ids)
        # 重置成功计数器
        self._goal_reached[env_ids] = 0

        # 完全按照CaptureXY：使用课程学习逻辑
        if self._task_parameters.spawn_curriculum:
            if step < self._task_parameters.spawn_curriculum_warmup:
                rmax = self._task_parameters.spawn_curriculum_max_dist
                rmin = self._task_parameters.spawn_curriculum_min_dist
            elif step > self._task_parameters.spawn_curriculum_end:
                rmax = self._task_parameters.spawn_radius_max
                rmin = self._task_parameters.spawn_radius_min
            else:
                progress = (step - self._task_parameters.spawn_curriculum_warmup) / (
                    self._task_parameters.spawn_curriculum_end
                    - self._task_parameters.spawn_curriculum_warmup
                )
                rmax = (
                    progress
                    * (
                        self._task_parameters.spawn_radius_max
                        - self._task_parameters.spawn_curriculum_max_dist
                    )
                    + self._task_parameters.spawn_curriculum_max_dist
                )
                rmin = (
                    progress
                    * (
                        self._task_parameters.spawn_radius_min
                        - self._task_parameters.spawn_curriculum_min_dist
                    )
                    + self._task_parameters.spawn_curriculum_min_dist
                )
        else:
            rmax = self._task_parameters.spawn_radius_max
            rmin = self._task_parameters.spawn_radius_min

        # 完全按照CaptureXY：随机化生成位置
        r = torch.rand((num_resets,), device=self._device) * (rmax - rmin) + rmin
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi

        # 完全按照CaptureXY：使用+=操作，将随机偏移添加到基准位置
        # 基准位置是initial_position（环境原点），加上随机偏移，再加上目标位置偏移
        initial_position[env_ids, 0] += (r) * torch.cos(theta) + self._berth_centers[env_ids, 0]
        initial_position[env_ids, 1] += (r) * torch.sin(theta) + self._berth_centers[env_ids, 1]
        initial_position[env_ids, 2] += 0

        print(f"get_spawns: 环境 {env_ids} 生成位置范围: r=[{rmin:.2f}, {rmax:.2f}]")
        print(f"get_spawns: 船位中心: {self._berth_centers[env_ids]}")
        print(f"get_spawns: 最终生成位置: x=[{initial_position[env_ids, 0].min():.2f}, {initial_position[env_ids, 0].max():.2f}], y=[{initial_position[env_ids, 1].min():.2f}, {initial_position[env_ids, 1].max():.2f}]")

        # 完全按照CaptureXY：随机化航向
        random_orient = torch.rand(num_resets, device=self._device) * math.pi
        initial_orientation[env_ids, 0] = torch.cos(random_orient * 0.5)
        initial_orientation[env_ids, 3] = torch.sin(random_orient * 0.5)
        return initial_position, initial_orientation

    def update_kills(self, step) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not."""

        # 检查成功条件
        success = self._check_success()
        self._goal_reached += success.int()
        
        # 检查终止条件
        die = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        
        # 成功后在容差内停留足够时间
        success_timeout = self._goal_reached >= self._task_parameters.kill_after_n_steps_in_tolerance
        die = die | success_timeout
        
        # 碰撞终止
        die = die | self.collision_detected
        
        # 距离太远终止
        too_far = self.position_dist > self._task_parameters.kill_dist
        die = die | too_far
        
        # 时间限制
        timeout = step >= self._task_parameters.time_limit
        die = die | timeout
        
        return die

    def generate_target(self, path, position):
        """
        参考CaptureXY的实现方式，只为env_0创建船位中心标记。
        Isaac Sim的环境克隆机制会自动将标记复制到其他环境。
        
        注意：在泊船任务中，我们需要确保可视化标记位置与计算位置一致。
        """
        
        print(f"=== generate_target 被调用 ===")
        print(f"path: {path}, position: {position}")
        
        # 重要：在泊船任务中，我们需要确保可视化标记位置与计算位置一致
        # 由于get_goals可能还没有被调用，我们暂时使用传入的position参数
        # 但会在get_goals被调用后，通过marker.set_world_poses更新到正确位置
        
        # 生成船位中心标记（绿色）
        color = torch.tensor([0, 1, 0])  # 绿色
        ball_radius = 0.2
        poll_radius = 0.02
        poll_length = 1.5
        
        VisualPin(
            prim_path=path + "/berth_center",
            translation=position,
            name="berth_center_0",
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            color=color,
        )
        
        print(f"为env_0创建船位中心标记: {path}/berth_center at {position}")
        print(f"注意：实际位置将在get_goals调用后通过marker.set_world_poses更新")
        print(f"=== generate_target 完成 ===")

    def generate_berth(self, path):
        """
        Generates visual markers for the berth walls.
        绘制矩形泊船位，一面开口，三面围挡。
        使用分段线绘制：开口边为绿色，其余三边为红色；角点用 VisualPin。
        
        暂时屏蔽，用于测试船位中心标识。
        """
        
        print("=== generate_berth 被调用，绘制船位矩形 ===")
        
        # 基于本地坐标系生成船位边界（每个 env 原点为中心，便于在场景创建阶段稳定显示）
        half_width = self._task_parameters.berth_width / 2
        half_length = self._task_parameters.berth_length / 2
        base_corners = [
            (-half_width, -half_length),  # [0] 左下角（后墙）
            ( half_width, -half_length),  # [1] 右下角（右墙）
            ( half_width,  half_length),  # [2] 右上角（开口）
            (-half_width,  half_length),  # [3] 左上角（左墙）
        ]
        
        print(f"船位尺寸: 宽度={self._task_parameters.berth_width}, 长度={self._task_parameters.berth_length}")
        print(f"基础角点: {base_corners}")
        
        # 创建船位边界标记
        for i in range(self._num_envs):
            # 逐边绘制：0-1(底部，红)，1-2(右侧，红)，2-3(顶部开口，绿)，3-0(左侧，红）
            edges = [
                (0, 1, (0.7, 0.0, 0.0)),  # bottom - red
                (1, 2, (0.7, 0.0, 0.0)),  # right - red
                (2, 3, (0.0, 0.7, 0.0)),  # top (opening) - green
                (3, 0, (0.7, 0.0, 0.0)),  # left - red
            ]
            for ei, ej, color in edges:
                prim_path_edge = f"/World/envs/env_{i}/berth_edge_{ei}_{ej}"
                self._draw_edge_segment(
                    prim_path_edge,
                    [base_corners[ei][0], base_corners[ei][1], 0.0],
                    [base_corners[ej][0], base_corners[ej][1], 0.0],
                    color=color,
                    width=0.02,
                )
            
            # [0] 左下角（深红色）- 后墙
            VisualPin(
                prim_path=f"/World/envs/env_{i}/berth_back_corner",
                translation=[base_corners[0][0], base_corners[0][1], 0.0],
                name="berth_back_corner",
                ball_radius=0.05,  # 更小的球
                poll_radius=0.005,  # 更细的杆
                poll_length=0.3,  # 更短的杆
                color=torch.tensor([0.7, 0, 0]),  # 深红色
            )
            
            # [1] 右下角（深红色）- 右墙
            VisualPin(
                prim_path=f"/World/envs/env_{i}/berth_right_corner",
                translation=[base_corners[1][0], base_corners[1][1], 0.0],
                name="berth_right_corner",
                ball_radius=0.05,  # 更小的球
                poll_radius=0.005,  # 更细的杆
                poll_length=0.3,  # 更短的杆
                color=torch.tensor([0.7, 0, 0]),  # 深红色
            )
            
            # [2] 右上角（绿色）- 开口
            VisualPin(
                prim_path=f"/World/envs/env_{i}/berth_front_corner",
                translation=[base_corners[2][0], base_corners[2][1], 0.0],
                name="berth_front_corner",
                ball_radius=0.05,  # 更小的球
                poll_radius=0.005,  # 更细的杆
                poll_length=0.3,  # 更短的杆
                color=torch.tensor([0, 0.7, 0]),  # 深绿色
            )
            
            # [3] 左上角（深红色）- 左墙
            VisualPin(
                prim_path=f"/World/envs/env_{i}/berth_left_corner",
                translation=[base_corners[3][0], base_corners[3][1], 0.0],
                name="berth_left_corner",
                ball_radius=0.05,  # 更小的球
                poll_radius=0.005,  # 更细的杆
                poll_length=0.3,  # 更短的杆
                color=torch.tensor([0.7, 0, 0]),  # 深红色
            )
        
        print(f"=== generate_berth 完成，为 {self._num_envs} 个环境创建了船位矩形 ===")

    def _draw_edge_segment(self, prim_path: str, p0, p1, color=(0.7, 0.0, 0.0), width=0.02):
        """在 USD 场景中以 BasisCurves 绘制一条线段 p0->p1。"""
        stage = omni.usd.get_context().get_stage()
        curve_prim = UsdGeom.BasisCurves.Define(stage, prim_path)
        curve_prim.CreateTypeAttr("linear")
        pts = [
            Gf.Vec3f(float(p0[0]), float(p0[1]), float(p0[2])),
            Gf.Vec3f(float(p1[0]), float(p1[1]), float(p1[2])),
        ]
        curve_prim.CreatePointsAttr(pts)
        curve_prim.CreateCurveVertexCountsAttr([2])
        curve_prim.CreateWidthsAttr([float(width), float(width)])
        curve_prim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

    def add_visual_marker_to_scene(self, scene):
        """
        Adds the visual markers to the scene.
        仿照 dynamic_position 的实现方式。
        现在添加完整的泊船可视化：船位中心、边线和角点。
        """
        
        print(f"=== add_visual_marker_to_scene 被调用 ===")

        # 添加船位中心标记
        try:
            berth_centers = XFormPrimView(prim_paths_expr="/World/envs/.*/berth_center")
            scene.add(berth_centers)
            print(f"成功添加船位中心标记到场景")
        except Exception as e:
            print(f"错误: 无法添加船位中心标记到场景: {e}")
            berth_centers = None
        
        # 添加船位边界标记（边线和角点）
        try:
            berth_corners = XFormPrimView(prim_paths_expr="/World/envs/.*/berth_.*_corner", name="berth_corners_view")
            scene.add(berth_corners)
            print(f"成功添加船位角点标记到场景")
        except Exception as e:
            print(f"警告: 无法添加船位角点标记到场景: {e}")
            berth_corners = None
        
        # 添加船位边线标记
        try:
            berth_edges = XFormPrimView(prim_paths_expr="/World/envs/.*/berth_edge_.*", name="berth_edges_view")
            scene.add(berth_edges)
            print(f"成功添加船位边线标记到场景")
        except Exception as e:
            print(f"警告: 无法添加船位边线标记到场景: {e}")
            berth_edges = None
        
        print(f"=== add_visual_marker_to_scene 完成 ===")
        return scene, berth_centers


