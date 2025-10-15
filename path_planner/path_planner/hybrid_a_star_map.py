from dataclasses import dataclass
from math import pi
from pathlib import Path as fPath
from typing import List, Tuple

import numpy as np
from dynamic_programming_heuristic import ANodeProto
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from utils import A, C, Pose2D, S


@dataclass
class HNode:
    x_index: int
    y_index: int
    yaw_index: int
    direction: bool
    x_list: List[float]
    y_list: List[float]
    yaw_list: List[float]
    directions: List[bool]
    steer: float = 0.0
    parent_index: int | None = None
    cost: float = 0

    def __repr__(self):
        return f"Node({self.x_index},{self.y_index},{self.yaw_index})"


class HMap:
    """
    * 内部地图栅格坐标系：
    使用numpy二维数组存储地图，考虑一个正常的矩阵写法，
    右上角[0,0]定为原点，x轴指向右方，y轴指向下方 (ROS的地图消息好像也是这个坐标系)，
    因此在提取某一坐标处地图信息时，需要使用map[y,x]的形式进行索引
    存在两套数值体系，int的栅格索引和float的连续坐标，原点相同

    * 与系统整体的地图坐标系的转换：
    类内部处理使用前述栅格坐标系，因此在在接收其他模块输入输出时，需要额外进行SE2转换
    TODO: 需要确认这个xy颠倒ROS信息是怎么表示
    """

    def __init__(
        self,
        ob_map: NDArray[np.bool_],
        resolution: float = A.XY_GRID_RESOLUTION,
        yaw_resolution: float = A.YAW_GRID_RESOLUTION,
        rr: float = C.BUBBLE_R,
        origin: Pose2D = Pose2D(0, 0, 0),
    ) -> None:
        """
        Args:
            ob_map (NDArray[np.bool_]): 二维数组，True表示有障碍物
            resolution (float, optional): 地图分辨率 [m]
            yaw_resolution (float): 偏航角的分辨率 [rad]
            rr (float, optional): 巡视器安全半径 [m]
            origin (float, optional): 接收到的地图中00栅格相对于地图坐标系的位姿
        """
        # 输入输出要加这个转换
        #! 仅考虑了平移，未对旋转加以测试
        self.origin_pose = origin
        self.SE2 = origin.SE2  # 内部 -> 外部
        self.SE2inv = origin.SE2inv  # 外部 -> 内部
        self.map_yaw = origin.yaw_rad  # 内部 -> 外部（加的话）

        self.kdTree: cKDTree | None = None
        self.resolution = resolution
        self.yaw_resolution = yaw_resolution
        self.rr = rr
        self.obstacle_map = self.add_manual_ob(ob_map)
        self.edf_map: NDArray[np.float64] = (
            distance_transform_edt(~self.obstacle_map) * resolution
        )  # [m] # type: ignore
        self.euclidean_dilated_ob_map: NDArray[np.bool_] = self.edf_map <= rr * A.SAFETY_MARGIN_RATIO  # 根据半径膨胀

        # 地图参数
        self.max_y_index, self.max_x_index = self.obstacle_map.shape  # 先y后x
        self.min_x_index, self.min_y_index = 0, 0
        self.x_width, self.y_width = self.obstacle_map.shape
        self.min_yaw_index = round(-pi / yaw_resolution) - 1
        self.max_yaw_index = round(pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw_index - self.min_yaw_index)

    def add_manual_ob(self, ob_map: NDArray[np.bool_]) -> NDArray[np.bool_]:
        res = self.resolution
        h, w = ob_map.shape
        map_size_x = w * res
        map_size_y = h * res

        # -------------------------------
        # A. QUICK_OB: 基于10等分区块编号
        # -------------------------------
        for gx, gy, r_cm in A.QUICK_OB:
            # 将编号映射到地图坐标（0~5m）
            x_m = (gx - 0.5) * 0.5
            y_m = (gy - 0.5) * 0.5
            x_m = x_m - 0.2
            y_m = -y_m + 0.9
            r_m = r_cm / 100.0  # cm → m
            self._draw_circle(ob_map, x_m, y_m, r_m)

        # -------------------------------
        # B. PRECISE_OB: 基于真实坐标
        # -------------------------------
        for x_m, y_m, r_cm in A.PRECISE_OB:
            x_m = x_m - 0.2
            y_m = -y_m + 0.9
            r_m = r_cm / 100.0
            self._draw_circle(ob_map, x_m, y_m, r_m)

        return ob_map

    def _draw_circle(self, ob_map: np.ndarray, x_m: float, y_m: float, r_m: float) -> None:
        """在地图上画一个圆形障碍（单位：米）"""
        out = np.array([[x_m], [y_m], [1]])
        inn = self.SE2inv @ out
        h, w = ob_map.shape
        res = self.resolution

        # 转换为像素坐标
        cx = int(round(inn[0] / res))
        cy = int(round(inn[1] / res))
        r_px = int(round(r_m / res))

        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_px**2
        ob_map[mask] = True

    @classmethod
    def from_file(cls, file: str | fPath):
        map = np.load(file)
        return cls(map)

    def verify_index(self, node: HNode):
        x_ind, y_ind = node.x_index, node.y_index
        if self.min_x_index <= x_ind <= self.max_x_index and self.min_y_index <= y_ind <= self.max_y_index:
            return True
        return False

    def calc_index(self, node: HNode):
        "计算扁平化的索引"
        return (
            (node.yaw_index - self.min_yaw_index) * self.x_width * self.y_width
            + (node.y_index - self.min_y_index) * self.x_width
            + (node.x_index - self.min_x_index)
        )

    def calc_index_2d(self, node: ANodeProto) -> int:
        "计算扁平化的索引，只考虑xy，A*启发项需要用这个索引"
        return (node.y_index - self.min_y_index) * self.x_width + (node.x_index - self.min_x_index)

    def world_to_map(self, x: float, y: float, yaw: float) -> Tuple[int, int, int]:
        return round(x / self.resolution), round(y / self.resolution), round(yaw / self.yaw_resolution)

    def world_to_map_2d(self, x: float, y: float) -> Tuple[int, int]:
        return round(x / self.resolution), round(y / self.resolution)

    def build_kdtree(self) -> cKDTree:
        """从 obstacle_map 中提取障碍物坐标，并构建 KDTree"""
        if self.kdTree is None:
            obstacle_indices = np.argwhere(self.obstacle_map)  # shape (N, 2)
            self.kd_tree_points = obstacle_indices * self.resolution  # 将地图索引转换为世界坐标（米）
            self.kd_tree_points = self.kd_tree_points[:, [1, 0]]  # 地图的索引是yx顺序的，进行交换
            self.kdTree = cKDTree(self.kd_tree_points)
        return self.kdTree

    def compute_point_distance_field(self, gx: float, gy: float) -> NDArray[np.floating]:
        """计算到给定一点的距离场(极坐标距离值)，输入单位为m，输出单位为px"""
        H, W = self.obstacle_map.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        dx = x - gx / self.resolution
        dy = y - gy / self.resolution
        return np.sqrt(dx**2 + dy**2)
