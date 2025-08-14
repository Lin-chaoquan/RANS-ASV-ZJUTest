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
    Parameters for the Berthing task."""

    # 泊船位参数
    berth_width: float = 4.0  # 船位宽度
    berth_length: float = 6.0  # 船位长度
    berth_center: list = None  # 船位中心 [x, y]
    
    # 碰撞检测参数
    collision_radius: float = 0.5  # 船体安全半径
    wall_distance_threshold: float = 0.8  # 墙距离阈值
    
    # 成功条件
    position_tolerance: float = 0.2  # 位置容差
    heading_tolerance: float = 0.1  # 航向容差
    time_limit: int = 1000  # 时间限制
    
    # 生成参数
    spawn_radius_min: float = 3.0  # 最小生成半径
    spawn_radius_max: float = 8.0  # 最大生成半径
    spawn_angle_range: float = 180  # 生成角度范围（度）
    
    # 终止条件
    goal_random_position: float = 0.0
    kill_after_n_steps_in_tolerance: int = 50  # 成功后在容差内停留步数
    kill_dist: float = 15.0  # 杀死距离
    boundary_cost: float = 50.0  # 边界惩罚
    goal_reward: float = 1000.0  # 成功奖励
    time_reward: float = -0.1  # 时间惩罚
    
    # 课程学习参数
    spawn_curriculum: bool = False
    spawn_curriculum_min_dist: float = 2.0
    spawn_curriculum_max_dist: float = 5.0
    spawn_curriculum_kill_dist: float = 20.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 1000

    def __post_init__(self) -> None:
        """
        Checks that the curriculum parameters are valid."""
        if self.berth_center is None:
            self.berth_center = [0.0, 0.0]
            
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