from dataclasses import dataclass
from math import pi
from pathlib import Path as fPath
from typing import List, Tuple

import cv2
import numpy as np
from dynamic_programming_heuristic import ANodeProto
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, uniform_filter
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
    cost: float = 0.0
    slope_cost: float = 0.0
    rough_cost: float = 0.0

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
    类内部处理使用前述栅格坐标系，因此在在接收其他模块输入输出时，需要额外进行SE2转换，目前仅考虑平移
    TODO: 需要确认这个xy颠倒ROS信息是怎么表示
    """

    def __init__(
        self,
        dem_map: NDArray[np.floating],
        resolution: float = A.XY_GRID_RESOLUTION,
        yaw_resolution: float = A.YAW_GRID_RESOLUTION,
        rr: float = C.BUBBLE_R,
        origin: Pose2D = Pose2D(0, 0, 0),
    ) -> None:
        """
        Args:
            ob_map (NDArray[np.floating]): 二维数组，数字高程图
            resolution (float, optional): 地图分辨率 [m]
            yaw_resolution (float): 偏航角的分辨率 [rad]
            rr (float, optional): 巡视器安全半径 [m]
            origin (float, optional): 接收到的地图中00栅格相对于地图坐标系的位姿
        """
        self.resolution = resolution
        self.dem_map = dem_map
        self.slope_map, self.obstacle_map, self.rough_map = self._calculate_slope_passable(dem_map)
        self.edf_map: NDArray[np.float64] = (
            distance_transform_edt(~self.obstacle_map) * resolution
        )  # [m] # type: ignore
        self.euclidean_dilated_ob_map: NDArray[np.bool_] = self.edf_map <= rr * A.SAFETY_MARGIN_RATIO  # 根据半径膨胀

        self.kdTree: cKDTree | None = None
        self.yaw_resolution = yaw_resolution
        self.rr = rr

        # 地图参数
        self.max_y_index, self.max_x_index = self.obstacle_map.shape  # 先y后x
        self.min_x_index, self.min_y_index = 0, 0
        self.x_width, self.y_width = self.obstacle_map.shape
        self.min_yaw_index = round(-pi / yaw_resolution) - 1
        self.max_yaw_index = round(pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw_index - self.min_yaw_index)

        # 输入输出要加这个转换
        #! 仅考虑了平移，未对旋转加以测试
        self.origin_pose = origin
        self.SE2 = origin.SE2  # 内部 -> 外部
        self.SE2inv = origin.SE2inv  # 外部 -> 内部
        self.map_yaw = origin.yaw_rad  # 内部 -> 外部（加的话）

    def _calculate_slope_passable(
        self, dem: NDArray
    ) -> Tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.floating]]:
        """根据输入高程图计算各种地图"""
        # *1.滤波
        dem_f = dem.astype(dtype=np.float32)
        if A.FILTER_SWITH_ON:
            dem_filtered = cv2.ximgproc.guidedFilter(guide=dem_f, src=dem_f, radius=A.FILTER_RADIUS, eps=A.FILTER_EPS)
        else:
            dem_filtered = dem_f

        # *2.坡度图
        cell_size: float = self.resolution
        rows, cols = dem_filtered.shape
        grad_x = np.zeros((rows - 2, cols - 2))
        grad_y = np.zeros((rows - 2, cols - 2))

        # 遍历每个 3x3 窗口
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # 获取 3x3 窗口
                window = dem_filtered[i - 1 : i + 2, j - 1 : j + 2]

                grad_x[i - 1, j - 1] = (
                    (window[0, 2] + 2 * window[1, 2] + window[2, 1]) - (window[0, 0] + 2 * window[1, 0] + window[0, 1])
                ) / (8 * cell_size)

                grad_y[i - 1, j - 1] = (
                    (window[2, 0] + 2 * window[2, 1] + window[2, 2]) - (window[0, 0] + 2 * window[0, 1] + window[0, 2])
                ) / (8 * cell_size)

        # 计算坡向
        # aspect = np.arctan2(-grad_y, grad_x)
        # 计算坡度
        slope = np.sqrt(grad_x**2 + grad_y**2)
        slope_padded = np.pad(slope, pad_width=1, mode="constant", constant_values=0.0)  # 在外围扩充一圈为0的边缘

        # *3.限制坡度得到可通行地图（障碍物地图）
        passable_map: NDArray[np.bool_] = (slope_padded > np.deg2rad(C.MAX_PASSABLE_SLOPE)).astype(np.bool_)

        # *4.计算崎岖度地图
        window_size = int(C.BUBBLE_R // self.resolution) | 1  # 保证是奇数
        mean = uniform_filter(dem, size=window_size)
        mean_sq = uniform_filter(dem**2, size=window_size)
        var = mean_sq - mean**2
        rough_map = np.sqrt(np.maximum(var, 0))  # 局部标准差

        return slope_padded, passable_map, rough_map

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
