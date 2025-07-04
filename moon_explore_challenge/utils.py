import math
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class Settings:
    class Car:
        WB = 0.35  # 轴距（rear to front wheel）
        W = 0.4  # 车轮廓宽（width of car）
        LF = 0.6  # 后轴到前端（distance from rear to vehicle front end）
        LB = 0.2  # 后轴到后端（distance from rear to vehicle back end）
        MAX_STEER = 0.5  # 前轮最大转向角 [rad] maximum steering angle

        BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
        BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

        # 这个只是画图用的，主函数里没用到
        # 此处能发现路径的参考点是车辆后轴中心
        VRX = [LF, LF, -LB, -LB, LF]
        VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

    class AStar:
        # ==============================================================================
        # 地图的相关参数
        # ==============================================================================
        MAP_MAX_SIZE = 7  # 地图尺寸 [m]
        XY_GRID_RESOLUTION = 0.1  # 栅格分辨率 [m]
        YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # 节点偏航角分辨率 [rad]
        MOTION_RESOLUTION = 0.01  # 路径积分的分辨率 [m]
        N_STEER = 20  # 转向指令的生成数量

        # ==============================================================================
        # 损失计算的相关参数
        # ==============================================================================
        SB_PENALTY = 100.0  # 换向惩罚（非系数）
        BACK_PENALTY = 5.0  # 倒车惩罚系数
        STEER_CHANGE_PENALTY = 5.0  # 转向变化惩罚系数
        STEER_PENALTY = 1.0  # 转向惩罚系数
        H_WEIGHT = 5.0  # 启发项权重，用于在cost基础上附加启发项时

        # ==============================================================================
        # 解析扩张相关
        # ==============================================================================
        # * 启发项物理含义：预期到达的距离，单位 [m]
        H_HIGH = 5.0  # 此处规定两个启发项阈值，频率在他们之间时线性递减
        H_LOW = 0.5

        # * N为解析扩张频率，此处规定上下界
        N_MAX = math.ceil((H_HIGH - H_LOW) / 10 / XY_GRID_RESOLUTION)
        N_MIN = 1

        # * 在验证解析扩张得到的路径是否碰撞时，先拒绝掉代价过高的路径提高验证效率
        # 某一次循环中的cost列表 [20.00284761554539, 179.13205454838817, 180.59115325247083, ...]
        RS_COST_REJECTION_RATIO = 2  # 相对于最低代价的比例，超出的路径会被直接拒绝不进行验证

        # ==============================================================================
        # 路径平滑相关
        # ==============================================================================
        LEARN_RATE = 0.1
        ITERATIONS: int = 100  # 迭代次数
        WEIGHT_SMOOTH = 0.5
        WEIGHT_OBSTACLE = 1.0
        WEIGHT_FIDELITY = 0.2
        OBSTACLE_SIGMA = 1.0

        # ==============================================================================
        # 其他参数
        # ==============================================================================
        SAFETY_MARGIN_RATIO = 1.2  # 对机器人半径扩张增加安全性

    class Debug:
        use_profile = True
        use_profile = False

    C = Car()
    A = AStar()


S = Settings()
C = S.C
A = S.A


class PoseDiff(NamedTuple):
    dist: float  # 欧氏距离
    yaw_diff_deg: float  # 偏航角差绝对值(角度)
    yaw_diff_rad: float


class Pose2D:
    def __init__(self, x: float, y: float, yaw: float, deg: bool = False) -> None:
        """
        二维的位姿

        Args:
            x (float): 全局x坐标
            y (float): 全局y坐标
            yaw (float): 弧度制偏航角
            deg (bool): 如果为True，则使用角度制定义偏航角，默认为False
        """
        self.x = x
        self.y = y
        if deg:
            self._yaw = math.radians(yaw)
        else:
            self._yaw = yaw
        pi = math.pi
        self._yaw = (self._yaw + pi) % (2 * pi) - pi

    @classmethod
    def from_pose_msg(cls, x: float, y: float, qx: float, qy: float, qz: float, qw: float) -> "Pose2D":
        """
        从 ROS Pose 消息创建 Pose2D 实例
        这里不处理 Pose 消息的实例，因此参数是提取后的
        坐标系是slam的图像坐标系：右下前
        要先转换回前左上

        Args:
            ROS 2 中的 geometry_msgs.msg.Pose 消息中的必要参数

        Returns:
            Pose2D: 生成的二维位姿
        """
        R_ = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # webots中相机系：前左上；图像的坐标系：右下前
        camera_rotation = np.dot(Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), R_.T)
        yaw = Rotation.from_matrix(camera_rotation).as_euler("xyz")[2]
        # yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz")[2]
        return cls(x, y, yaw)

    def __sub__(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __xor__(self, other: "Pose2D") -> PoseDiff:
        distance = self - other
        diff_abs = abs(self.yaw_deg360 - other.yaw_deg360)
        yaw_diff = min(diff_abs, 360 - diff_abs)
        return PoseDiff(dist=distance, yaw_diff_deg=yaw_diff, yaw_diff_rad=yaw_diff * math.pi / 180)

    @property
    def t(self) -> NDArray[np.float64]:
        return np.array([[self.x], [self.y]])

    @property
    def SO2(self) -> NDArray[np.float64]:
        "表示在全局坐标系"
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        return np.array([[cos_y, -sin_y], [sin_y, cos_y]])

    @property
    def SO2inv(self) -> NDArray[np.float64]:
        "表示在本体坐标系"
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        return np.array([[cos_y, sin_y], [-sin_y, cos_y]])

    @property
    def SE2(self) -> NDArray[np.float64]:
        "表示在全局坐标系"
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        return np.array([[cos_y, -sin_y, self.x], [sin_y, cos_y, self.y], [0, 0, 1]])

    @property
    def SE2inv(self) -> NDArray[np.float64]:
        "表示在本体坐标系"
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        return np.array(
            [
                [cos_y, sin_y, -cos_y * self.x - sin_y * self.y],
                [-sin_y, cos_y, sin_y * self.x - cos_y * self.y],
                [0, 0, 1],
            ]
        )

    @property
    def xy(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y])

    @property
    def yaw_rad(self, deg=False) -> float:
        "偏航角，弧度制"
        return self._yaw

    @yaw_rad.setter
    def yaw_rad(self, value: float) -> None:
        self._yaw = value

    @property
    def yaw_deg180(self) -> float:
        """返回 [-180, 180) 的角度值"""
        deg = math.degrees(self._yaw) % 360
        return (deg + 180) % 360 - 180

    @property
    def yaw_deg360(self) -> float:
        """返回 [0, 360) 的角度值"""
        return math.degrees(self._yaw) % 360

    def __str__(self):
        return f"{self.x}, {self.y}, {self.yaw_deg360}"

    def __repr__(self):
        return f"{self.x}, {self.y}, {self.yaw_deg360}"


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if isinstance(x, list):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(
            x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width
        )
        plt.plot(x, y)


def plot_car(x, y, yaw):
    car_color = "-k"
    c, s = math.cos(yaw), math.sin(yaw)
    rot = Pose2D(0, 0, -yaw).SO2
    car_outline_x, car_outline_y = [], []
    VRX, VRY = Settings.Car.VRX, Settings.Car.VRY
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)
