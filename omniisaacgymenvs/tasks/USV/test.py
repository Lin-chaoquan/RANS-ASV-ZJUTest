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
        
        # 参考 CaptureXY：为每个环境生成各自的船位中心（一次性初始化）
        random_range = float(getattr(self._task_parameters, "goal_random_position", 0.0))
        if random_range <= 0.0:
            random_range = 2.0  # 合理默认值，避免所有中心重合
        self._berth_centers = (
            torch.rand((self._num_envs, 2), device=self._device) * 2.0 - 1.0
        ) * random_range
        # 基于船位中心计算四个角点
        self._calculate_berth_corners()

        # 初始化统计变量
        self.prev_position_dist = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self.position_dist = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self.heading_error = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self.combined_reward = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self.collision_detected = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._position_error = torch.zeros(self._num_envs, 2, device=self._device, dtype=torch.float32)

    def _calculate_berth_corners(self, env_ids=None):
        """计算船位四个角点（逆时针）
        船位布局：
        [3] 左上角 -------- [2] 右上角 (开口)
         |                    |
         |                    |
        [0] 左下角 -------- [1] 右下角
        """
        half_width = self._task_parameters.berth_width / 2
        half_length = self._task_parameters.berth_length / 2
        
        # 四个角点（逆时针，从左下开始）
        base_corners = torch.tensor([
            [-half_width, -half_length],  # [0] 左下角
            [half_width, -half_length],   # [1] 右下角
            [half_width, half_length],    # [2] 右上角 (开口)
            [-half_width, half_length]    # [3] 左上角
        ], device=self._device, dtype=torch.float32)
        
        if env_ids is None:
            # 为所有环境计算
            for i in range(self._num_envs):
                self._berth_corners[i] = base_corners + self._berth_centers[i]
        else:
            # 只为指定环境计算
            for env_id in env_ids:
                self._berth_corners[env_id] = base_corners + self._berth_centers[env_id]

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
        Computes the observation tensor from the current state of the robot."""

        # 计算到船位中心的距离
        self._position_error = self._berth_centers - current_state["position"][:, :2]
        
        # 计算航向误差
        theta = torch.atan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        # 计算目标航向（朝向船位中心）
        beta = torch.atan2(self._position_error[:, 1], self._position_error[:, 0])
        # 计算航向误差
        alpha = torch.fmod(beta - theta + math.pi, 2 * math.pi) - math.pi
        self.heading_error = torch.abs(alpha)
        
        # 更新任务数据
        self._task_data[:, 0] = torch.cos(alpha)
        self._task_data[:, 1] = torch.sin(alpha)
        self._task_data[:, 2] = torch.norm(self._position_error, dim=1)
        
        # 检查碰撞
        self.collision_detected = self._check_collision(
            current_state["position"][:, :2]
        )
        
        return self.update_observation_tensor(current_state, observation_frame)

    def _check_collision(self, usv_position):
        """检查USV与船位的碰撞"""
        collision = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        
        # 为每个环境检查碰撞
        for env_id in range(self._num_envs):
            # 检查与每条边的距离
            for i in range(4):
                wall_start = self._berth_corners[env_id, i]
                wall_end = self._berth_corners[env_id, (i + 1) % 4]
                
                # 计算到墙的距离
                wall_distance = self._point_to_line_distance(
                    usv_position[env_id:env_id+1], 
                    wall_start.unsqueeze(0), 
                    wall_end.unsqueeze(0)
                )
                
                # 如果距离小于碰撞半径，判定为碰撞
                if wall_distance < self._task_parameters.collision_radius:
                    collision[env_id] = True
                    break
        
        return collision

    def _generate_berth_centers(self, env_ids):
        """为指定环境生成船位中心"""
        # 按需为指定环境重新随机中心（保持与 CaptureXY 一致的随机逻辑）
        random_range = float(getattr(self._task_parameters, "goal_random_position", 0.0))
        if random_range <= 0.0:
            random_range = 2.0
        num_resets = len(env_ids)
        self._berth_centers[env_ids] = (
            torch.rand((num_resets, 2), device=self._device) * 2.0 - 1.0
        ) * random_range

    def _point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的距离"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length = torch.norm(line_vec, dim=-1)
        line_length = torch.clamp(line_length, min=1e-6)  # 避免除零
        
        # 计算投影参数
        t = torch.sum(point_vec * line_vec, dim=-1) / (line_length ** 2)
        t = torch.clamp(t, 0, 1)
        
        # 最近点
        closest_point = line_start + t.unsqueeze(-1) * line_vec
        
        return torch.norm(point - closest_point, dim=-1)

    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot."""

        # 计算奖励
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        self.combined_reward = self._reward_parameters.compute_reward(
            current_state,
            actions,
            self.position_dist,
            self.heading_error,
            self._berth_corners,
            self._task_parameters.collision_radius,
        )

        # 重置时奖励为0
        self.combined_reward[self.just_had_been_reset] = 0
        self.just_had_been_reset = torch.tensor(
            [], device=self._device, dtype=torch.long
        )

        # 检查是否成功
        success = self._check_success()
        goal_reward = success * self._task_parameters.goal_reward
        
        # 时间惩罚
        time_reward = self._task_parameters.time_reward

        return self.combined_reward + goal_reward + time_reward

    def _check_success(self):
        """检查是否成功泊船"""
        position_success = self.position_dist < self._task_parameters.position_tolerance
        heading_success = self.heading_error < self._task_parameters.heading_tolerance
        success = position_success & heading_success
        
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
        Resets the goal_reached_flag when an agent manages to solve its task."""

        self._goal_reached[env_ids] = 0
        self.just_had_been_reset = env_ids.clone()
        
        # 为重置的环境生成新的船位中心
        self._generate_berth_centers(env_ids)
        
        # 更新船位角点
        self._calculate_berth_corners(env_ids)

    def get_goals(
        self,
        env_ids: torch.Tensor,
        targets_position: torch.Tensor,
        targets_orientation: torch.Tensor,
    ) -> list:
        """
        Generates the berth center as the goal."""

        num_goals = len(env_ids)
        # 目标位置就是船位中心
        targets_position[env_ids, :2] = self._berth_centers[env_ids]
        return targets_position, targets_orientation

    def get_spawns(
        self,
        env_ids: torch.Tensor,
        initial_position: torch.Tensor,
        initial_orientation: torch.Tensor,
        step: int = 0,
    ) -> list:
        """
        Generates spawning positions for the robots around the berth.
        按照 CaptureXY 的方法：半径在[min,max]均匀采样，角度在[0, 2π]均匀采样。"""

        num_resets = len(env_ids)
        # 重置成功计数器
        self._goal_reached[env_ids] = 0

        # 课程学习（与 CaptureXY 相同逻辑，但使用泊位参数命名）
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

        # CaptureXY 风格的均匀采样：半径 + 全向角度
        r = torch.rand((num_resets,), device=self._device) * (rmax - rmin) + rmin
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi

        # 以船位中心为圆心，均匀环形分布
        initial_position[env_ids, 0] = self._berth_centers[env_ids, 0] + r * torch.cos(theta)
        initial_position[env_ids, 1] = self._berth_centers[env_ids, 1] + r * torch.sin(theta)
        initial_position[env_ids, 2] = 0

        # 随机化航向
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
        Generates a visual marker for the berth center.
        仿照 dynamic_position 的实现方式。
        """

        color = torch.tensor([0, 1, 0])  # 绿色
        ball_radius = 0.3
        poll_radius = 0.025
        poll_length = 2
        VisualPin(
            prim_path=path + "/berth_center",
            translation=position,
            name="berth_center0",
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            color=color,
        )

    def generate_berth(self, path):
        """
        Generates visual markers for the berth walls.
        绘制矩形泊船位，一面开口，三面围挡。
        使用分段线绘制：开口边为绿色，其余三边为红色；角点用 VisualPin。
        """
        
        # 基于本地坐标系生成船位边界（每个 env 原点为中心，便于在场景创建阶段稳定显示）
        half_width = self._task_parameters.berth_width / 2
        half_length = self._task_parameters.berth_length / 2
        base_corners = [
            (-half_width, -half_length),  # [0] 左下角（后墙）
            ( half_width, -half_length),  # [1] 右下角（右墙）
            ( half_width,  half_length),  # [2] 右上角（开口）
            (-half_width,  half_length),  # [3] 左上角（左墙）
        ]
        
        # 创建船位边界标记
        for i in range(self._num_envs):
            # 逐边绘制：0-1(底部，红)，1-2(右侧，红)，2-3(顶部开口，绿)，3-0(左侧，红)
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
        """

        # 添加船位中心标记
        berth_centers = XFormPrimView(prim_paths_expr="/World/envs/.*/berth_center")
        scene.add(berth_centers)
        
        # 暂时注释掉矩形框的可视化
        # 添加船位矩形框标记
        # try:
        #     berth_rectangles = XFormPrimView(prim_paths_expr="/World/envs/.*/berth_rectangle", name="berth_rectangles_view")
        #     scene.add(berth_rectangles)
        # except Exception as e:
        #     print(f"Warning: Could not add berth rectangles to scene: {e}")
        #     berth_rectangles = None
        
        # 添加船位边界标记（如果存在）
        try:
            berth_corners = XFormPrimView(prim_paths_expr="/World/envs/.*/berth_.*_corner", name="berth_corners_view")
            scene.add(berth_corners)
        except Exception as e:
            print(f"Warning: Could not add berth corners to scene: {e}")
            berth_corners = None
        
        return scene, berth_centers


