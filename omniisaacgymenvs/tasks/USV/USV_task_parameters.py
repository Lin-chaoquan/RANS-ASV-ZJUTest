__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
__status__ = "development"

from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class CaptureXYParameters:
    """
    Parameters for the CaptureXY task."""

    position_tolerance: float = 0.1
    kill_after_n_steps_in_tolerance: int = 1
    goal_random_position: float = 0.0
    max_spawn_dist: float = 11
    min_spawn_dist: float = 0.5
    kill_dist: float = 20.0
    boundary_cost: float = 25
    goal_reward: float = 100.0
    time_reward: float = -0.1

    spawn_curriculum: bool = False
    spawn_curriculum_min_dist: float = 0.2
    spawn_curriculum_max_dist: float = 3.0
    spawn_curriculum_kill_dist: float = 30.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 1000

    def __post_init__(self) -> None:
        """
        Checks that the curicullum parameters are valid."""

        assert self.spawn_curriculum_mode.lower() in [
            "linear"
        ], "Linear is the only currently supported mode."
        if not self.spawn_curriculum:
            self.spawn_curriculum_max_dist = 0
            self.spawn_curriculum_min_dist = 0
            self.spawn_curriculum_kill_dist = 0
            self.spawn_curriculum_mode = 0
            self.spawn_curriculum_warmup = 0
            self.spawn_curriculum_end = 0


@dataclass
class GoToXYParameters:
    """
    Parameters for the GoToXY task."""

    position_tolerance: float = 0.1
    kill_after_n_steps_in_tolerance: int = 1
    goal_random_position: float = 0.0
    max_spawn_dist: float = 11
    min_spawn_dist: float = 0.5
    kill_dist: float = 20.0
    boundary_cost: float = 25
    goal_reward: float = 100.0
    time_reward: float = -0.1

    spawn_curriculum: bool = False
    spawn_curriculum_min_dist: float = 0.2
    spawn_curriculum_max_dist: float = 3.0
    spawn_curriculum_kill_dist: float = 30.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 1000

    def __post_init__(self) -> None:
        """
        Checks that the curicullum parameters are valid."""

        assert self.spawn_curriculum_mode.lower() in [
            "linear"
        ], "Linear is the only currently supported mode."
        if not self.spawn_curriculum:
            self.spawn_curriculum_max_dist = 0
            self.spawn_curriculum_min_dist = 0
            self.spawn_curriculum_kill_dist = 0
            self.spawn_curriculum_mode = 0
            self.spawn_curriculum_warmup = 0
            self.spawn_curriculum_end = 0


@dataclass
class GoToPoseParameters:
    """
    Parameters for the GoToPose task."""

    position_tolerance: float = 0.01
    heading_tolerance: float = 0.025
    kill_after_n_steps_in_tolerance: int = 500
    goal_random_position: float = 0.0
    max_spawn_dist: float = 3.0
    min_spawn_dist: float = 0.5
    kill_dist: float = 10.0

    spawn_curriculum: bool = False
    spawn_curriculum_min_dist: float = 0.5
    spawn_curriculum_max_dist: float = 2.5
    spawn_curriculum_kill_dist: float = 3.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 750

    def __post_init__(self) -> None:
        """
        Checks that the curicullum parameters are valid."""

        assert self.spawn_curriculum_mode.lower() in [
            "linear"
        ], "Linear is the only currently supported mode."
        if not self.spawn_curriculum:
            self.spawn_curriculum_max_dist = 0
            self.spawn_curriculum_min_dist = 0
            self.spawn_curriculum_kill_dist = 0
            self.spawn_curriculum_warmup = 0
            self.spawn_curriculum_end = 0


@dataclass
class KeepXYParameters:
    """
    Parameters for the KeepXY task."""

    position_tolerance: float = 0.1
    kill_after_n_steps_in_tolerance: int = 1
    goal_random_position: float = 0.0
    max_spawn_dist: float = 11
    min_spawn_dist: float = 0.5
    kill_dist: float = 20.0
    boundary_cost: float = 25
    goal_reward: float = 100.0
    time_reward: float = -0.1

    spawn_curriculum: bool = False
    spawn_curriculum_min_dist: float = 0.2
    spawn_curriculum_max_dist: float = 3.0
    spawn_curriculum_kill_dist: float = 30.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 1000

    def __post_init__(self) -> None:
        """
        Checks that the curicullum parameters are valid."""

        assert self.spawn_curriculum_mode.lower() in [
            "linear"
        ], "Linear is the only currently supported mode."
        if not self.spawn_curriculum:
            self.spawn_curriculum_max_dist = 0
            self.spawn_curriculum_min_dist = 0
            self.spawn_curriculum_kill_dist = 0
            self.spawn_curriculum_mode = 0
            self.spawn_curriculum_warmup = 0
            self.spawn_curriculum_end = 0


@dataclass
class TrackXYVelocityParameters:
    """
    Parameters for the TrackXYVelocity task."""

    lin_vel_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_velocity: float = 0.75
    kill_dist: float = 500.0


@dataclass
class TrackXYOVelocityParameters:
    """
    Parameters for the TrackXYOVelocity task."""

    lin_vel_tolerance: float = 0.01
    ang_vel_tolerance: float = 0.025
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_linear_velocity: float = 0.75
    goal_random_angular_velocity: float = 1
    kill_dist: float = 500.0


@dataclass
class DynamicPositionParameters:
    """
    Parameters for the TrackXYOVelocity task."""

    position_tolerance: float = 0.1
    kill_after_n_steps_in_tolerance: int = 1
    goal_random_position: float = 0.0
    max_spawn_dist: float = 11
    min_spawn_dist: float = 0.5
    kill_dist: float = 20.0
    boundary_cost: float = 25
    goal_reward: float = 100.0
    time_reward: float = -0.1

    spawn_curriculum: bool = False
    spawn_curriculum_min_dist: float = 0.2
    spawn_curriculum_max_dist: float = 3.0
    spawn_curriculum_kill_dist: float = 30.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 1000

    def __post_init__(self) -> None:
        """
        Checks that the curicullum parameters are valid."""

        assert self.spawn_curriculum_mode.lower() in [
            "linear"
        ], "Linear is the only currently supported mode."
        if not self.spawn_curriculum:
            self.spawn_curriculum_max_dist = 0
            self.spawn_curriculum_min_dist = 0
            self.spawn_curriculum_kill_dist = 0
            self.spawn_curriculum_mode = 0
            self.spawn_curriculum_warmup = 0
            self.spawn_curriculum_end = 0


@dataclass
class BerthingParameters:
    """
    改进的泊船任务参数，专门针对自动泊船任务优化。
    核心特性：
    1. 智能方向引导：鼓励从绿边（开口）进入
    2. 精确碰撞检测：区分红边（墙壁）和绿边（开口）
    3. 姿态稳定性：强烈抑制不必要的运动
    4. 渐进式任务：根据距离目标远近调整任务难度
    """

    # 泊船位参数
    berth_width: float = 4.0  # 船位宽度
    berth_length: float = 6.0  # 船位长度
    berth_center: list = None  # 船位中心 [x, y]
    
    # 碰撞检测参数
    collision_radius: float = 0.6  # 船体安全半径
    wall_distance_threshold: float = 1.0  # 墙距离阈值
    opening_distance_threshold: float = 0.8  # 开口距离阈值
    
    # 成功条件
    position_tolerance: float = 0.15  # 位置容差（更严格）
    heading_tolerance: float = 0.08  # 航向容差（更严格）
    time_limit: int = 1000  # 时间限制
    
    # 生成参数
    spawn_radius_min: float = 3.0  # 最小生成半径
    spawn_radius_max: float = 8.0  # 最大生成半径
    spawn_angle_range: float = 90.0  # 生成角度范围（度）- 限制在开口方向
    
    # 终止条件
    goal_random_position: float = 0.0
    kill_after_n_steps_in_tolerance: int = 100  # 成功后在容差内停留步数（更严格）
    kill_dist: float = 15.0  # 杀死距离
    boundary_cost: float = 100.0  # 边界惩罚（更严格）
    goal_reward: float = 1000.0  # 成功奖励
    time_reward: float = -0.1  # 时间惩罚
    
    # 方向引导参数
    entry_direction_weight: float = 1.5  # 进入方向权重
    entry_angle_tolerance: float = 0.3  # 进入角度容差（弧度）
    preferred_entry_distance: float = 2.0  # 首选进入距离
    
    # 姿态稳定性参数
    max_linear_velocity: float = 3.0  # 最大线速度
    max_angular_velocity: float = 1.0  # 最大角速度
    stability_threshold: float = 0.5  # 稳定性阈值
    
    # 课程学习参数
    spawn_curriculum: bool = True  # 启用课程学习
    spawn_curriculum_min_dist: float = 2.0
    spawn_curriculum_max_dist: float = 5.0
    spawn_curriculum_kill_dist: float = 20.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 1000
    
    # 渐进式任务参数
    distance_stages: list = None  # 距离阶段列表
    difficulty_multipliers: list = None  # 对应难度倍数

    def __post_init__(self) -> None:
        """
        初始化渐进式任务参数和验证参数有效性。
        """
        if self.berth_center is None:
            self.berth_center = [0.0, 0.0]
            
        # 初始化渐进式任务参数
        if self.distance_stages is None:
            self.distance_stages = [5.0, 3.0, 2.0, 1.0]  # 距离阶段
        if self.difficulty_multipliers is None:
            self.difficulty_multipliers = [0.5, 1.0, 1.5, 2.0]  # 对应难度倍数
            
        # 验证课程学习参数
        assert self.spawn_curriculum_mode.lower() in [
            "linear"
        ], "Linear is the only currently supported mode."
        if not self.spawn_curriculum:
            self.spawn_curriculum_max_dist = 0
            self.spawn_curriculum_min_dist = 0
            self.spawn_curriculum_kill_dist = 0
            self.spawn_curriculum_mode = 0
            self.spawn_curriculum_warmup = 0
            self.spawn_curriculum_end = 0
        
        # 验证泊船位参数
        assert self.berth_width > 0, "船位宽度必须大于0"
        assert self.berth_length > 0, "船位长度必须大于0"
        assert self.collision_radius > 0, "碰撞半径必须大于0"
        
        # 验证生成参数
        assert self.spawn_radius_min < self.spawn_radius_max, "最小生成半径必须小于最大生成半径"
        assert 0 < self.spawn_angle_range <= 360, "生成角度范围必须在0-360度之间"
        
        # 验证成功条件
        assert self.position_tolerance > 0, "位置容差必须大于0"
        assert self.heading_tolerance > 0, "航向容差必须大于0"
        assert self.time_limit > 0, "时间限制必须大于0"