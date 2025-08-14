__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from typing import Optional, Sequence
import numpy as np
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omniisaacgymenvs.utils.shape_utils import Pin
from pxr import Usd, UsdGeom, Gf
import torch
import math
import omni.usd

class VisualPin(XFormPrim, Pin):
    """_summary_

    Args:
        prim_path (str): _description_
        name (str, optional): _description_. Defaults to "visual_arrow".
        position (Optional[Sequence[float]], optional): _description_. Defaults to None.
        translation (Optional[Sequence[float]], optional): _description_. Defaults to None.
        orientation (Optional[Sequence[float]], optional): _description_. Defaults to None.
        scale (Optional[Sequence[float]], optional): _description_. Defaults to None.
        visible (Optional[bool], optional): _description_. Defaults to True.
        color (Optional[np.ndarray], optional): _description_. Defaults to None.
        radius (Optional[float], optional): _description_. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.

    Raises:
        Exception: _description_
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "visual_pin",
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        visible: Optional[bool] = True,
        color: Optional[np.ndarray] = None,
        ball_radius: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
    ) -> None:
        if visible is None:
            visible = True
        if visual_material is None:
            if color is None:
                color = np.array([0.5, 0.5, 0.5])
            visual_prim_path = find_unique_string_name(
                initial_name="/World/Looks/visual_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
        XFormPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
        )
        Pin.__init__(self, prim_path, ball_radius, poll_radius, poll_length)
        VisualPin.apply_visual_material(self, visual_material)
        self.setBallRadius(ball_radius)
        self.setPollRadius(poll_radius)
        self.setPollLength(poll_length)
        return


class FixedPin(VisualPin):
    """_summary_

    Args:
        prim_path (str): _description_
        name (str, optional): _description_. Defaults to "fixed_sphere".
        position (Optional[np.ndarray], optional): _description_. Defaults to None.
        translation (Optional[np.ndarray], optional): _description_. Defaults to None.
        orientation (Optional[np.ndarray], optional): _description_. Defaults to None.
        scale (Optional[np.ndarray], optional): _description_. Defaults to None.
        visible (Optional[bool], optional): _description_. Defaults to None.
        color (Optional[np.ndarray], optional): _description_. Defaults to None.
        radius (Optional[np.ndarray], optional): _description_. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.
        physics_material (Optional[PhysicsMaterial], optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "fixed_arrow",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        ball_radius: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
    ) -> None:
        if not is_prim_path_valid(prim_path):
            # set default values if no physics material given
            if physics_material is None:
                static_friction = 0.2
                dynamic_friction = 1.0
                restitution = 0.0
                physics_material_path = find_unique_string_name(
                    initial_name="/World/Physics_Materials/physics_material",
                    is_unique_fn=lambda x: not is_prim_path_valid(x),
                )
                physics_material = PhysicsMaterial(
                    prim_path=physics_material_path,
                    dynamic_friction=dynamic_friction,
                    static_friction=static_friction,
                    restitution=restitution,
                )
        VisualPin.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            visual_material=visual_material,
        )
        # XFormPrim.set_collision_enabled(self, True)
        # if physics_material is not None:
        #    FixedArrow.apply_physics_material(self, physics_material)
        return


class DynamicPin(RigidPrim, FixedPin):
    """_summary_

    Args:
        prim_path (str): _description_
        name (str, optional): _description_. Defaults to "dynamic_sphere".
        position (Optional[np.ndarray], optional): _description_. Defaults to None.
        translation (Optional[np.ndarray], optional): _description_. Defaults to None.
        orientation (Optional[np.ndarray], optional): _description_. Defaults to None.
        scale (Optional[np.ndarray], optional): _description_. Defaults to None.
        visible (Optional[bool], optional): _description_. Defaults to None.
        color (Optional[np.ndarray], optional): _description_. Defaults to None.
        radius (Optional[np.ndarray], optional): _description_. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.
        physics_material (Optional[PhysicsMaterial], optional): _description_. Defaults to None.
        mass (Optional[float], optional): _description_. Defaults to None.
        density (Optional[float], optional): _description_. Defaults to None.
        linear_velocity (Optional[Sequence[float]], optional): _description_. Defaults to None.
        angular_velocity (Optional[Sequence[float]], optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "dynamic_sphere",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        ball_radius: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        linear_velocity: Optional[Sequence[float]] = None,
        angular_velocity: Optional[Sequence[float]] = None,
    ) -> None:
        if not is_prim_path_valid(prim_path):
            if mass is None:
                mass = 0.02
        FixedPin.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            visual_material=visual_material,
            physics_material=physics_material,
        )
        RigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            mass=mass,
            density=density,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )

class VisualRectangleLine:
    """
    使用 USD BasisCurves 绘制矩形轮廓。
    参数
    - prim_path: USD 绝对路径，例如 /World/envs/env_0/rect
    - center: [x, y, z] 或 [x, y]（z 省略则默认为 0）
    - size: [length_x, length_y]，矩形的长宽
    - yaw: 绕 Z 轴的偏航角（弧度）
    - color: (r, g, b)，0~1
    - width: 线宽
    - device: torch 设备，用于张量计算（可选）
    """
    def __init__(
        self,
        prim_path: str,
        center,
        size,
        yaw: float = 0.0,
        color=(1.0, 0.0, 0.0),
        width: float = 2.0,
        device: torch.device = torch.device("cpu"),
        closed: bool = True,
    ):
        # 准备输入
        center = torch.tensor(center, dtype=torch.float32, device=device)
        if center.numel() == 2:
            center = torch.tensor([center[0], center[1], 0.0], dtype=torch.float32, device=device)

        size = torch.tensor(size, dtype=torch.float32, device=device)
        assert size.numel() == 2, "size 应为 [length_x, length_y]"
        hx, hy = size[0] * 0.5, size[1] * 0.5

        # 局部四个角点（逆时针）
        corners_local = torch.tensor(
            [
                [-hx, -hy, 0.0],
                [ hx, -hy, 0.0],
                [ hx,  hy, 0.0],
                [-hx,  hy, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # 绕 Z 轴旋转
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        corners_world = (corners_local @ R.T) + center  # [4,3]

        # 闭合
        if closed:
            corners_world = torch.vstack([corners_world, corners_world[0]])

        # 写入 USD BasisCurves
        stage = omni.usd.get_context().get_stage()
        curve_prim = UsdGeom.BasisCurves.Define(stage, prim_path)
        # 线型为折线
        curve_prim.CreateTypeAttr("linear")
        # 点集
        pts = corners_world.detach().cpu().numpy().tolist()
        curve_prim.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
        # 顶点计数（一个曲线）
        curve_prim.CreateCurveVertexCountsAttr([len(pts)])
        # 宽度（每点一个，避免 hydra 宽度插值警告）
        curve_prim.CreateWidthsAttr([float(width)] * len(pts))
        # 颜色
        display_color = [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        curve_prim.CreateDisplayColorAttr(display_color)

    @staticmethod
    def update(
        prim_path: str,
        center,
        size,
        yaw: float = 0.0,
        width: Optional[float] = None,
        color: Optional[Sequence[float]] = None,
        device: torch.device = torch.device("cpu"),
        closed: bool = True,
    ):
        """更新已存在矩形的 points/width/color。"""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim {prim_path} 不存在")

        center = torch.tensor(center, dtype=torch.float32, device=device)
        if center.numel() == 2:
            center = torch.tensor([center[0], center[1], 0.0], dtype=torch.float32, device=device)
        size = torch.tensor(size, dtype=torch.float32, device=device)
        hx, hy = size[0] * 0.5, size[1] * 0.5

        corners_local = torch.tensor(
            [
                [-hx, -hy, 0.0],
                [ hx, -hy, 0.0],
                [ hx,  hy, 0.0],
                [-hx,  hy, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        corners_world = (corners_local @ R.T) + center
        if closed:
            corners_world = torch.vstack([corners_world, corners_world[0]])

        pts = corners_world.detach().cpu().numpy().tolist()
        curve = UsdGeom.BasisCurves(prim)
        curve.GetPointsAttr().Set([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
        curve.GetCurveVertexCountsAttr().Set([len(pts)])

        if width is not None:
            curve.GetWidthsAttr().Set([float(width)] * len(pts))
        if color is not None:
            curve.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
