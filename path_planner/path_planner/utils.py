import math
import sys
from functools import cached_property
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class Settings:
    class Car:
        # 轴距210mm
        # 车轮廓宽375mm
        # 车轴到前端310mm
        # 车轴到后端90mm
        # 最大转向角25°
        # 最大转弯曲率2.22（这个和计算公式一致）
        # ? 后轴到中心139mm
        # ? 最小包络圆543mm

        WB = 0.21  # 轴距（rear to front wheel）
        W = 0.375  # 车轮廓宽（width of car）
        LF = 0.31  # 后轴到前端（distance from rear to vehicle front end）
        LB = 0.09  # 后轴到后端（distance from rear to vehicle back end）
        MAX_STEER = np.deg2rad(25.0)  # 前轮最大转向角 [rad] (maximum steering angle)
        MAX_C = math.tan(MAX_STEER) / WB  # 最大转弯曲率 [1/m]

        BUBBLE_DIST = (LF - LB) / 2.0  # 后轴到中心（distance from rear to center of vehicle.）
        BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # 车辆半径，以中心为圆心绘制最小包络圆形
        MAX_PASSABLE_SLOPE = 20  # [deg] 最大通行坡度

        # 这个只是画图用的，主函数里没用到
        # 此处能发现路径的参考点是车辆后轴中心
        VRX = [LF, LF, -LB, -LB, LF]
        VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

    class AStar:
        # ==============================================================================
        # 地图的相关参数
        # ==============================================================================
        XY_GRID_RESOLUTION = 0.1  # 栅格分辨率 [m] (这个只是离线测试的配置值，实际应用时会使用ROS消息中的分辨率)
        YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # 节点偏航角分辨率 [rad]
        MOTION_RESOLUTION_RATIO = 0.1  # 这个比率用于计算路径的采样分辨率，此变量描述相对于地图栅格分辨率的比例
        N_STEER = 20  # 转向指令的生成数量（节点扩展时，均匀生成20个不同角度的转向指令用于扩展节点）
        # OCCUPANCY_THRESHOLD = 65  # 占据栅格地图的概率阈值，大于此值表示为障碍物
        SAFETY_MARGIN_RATIO = 1.2  # 对机器人半径额外乘安全系数
        # SAFETY_MARGIN_RATIO = 0  # 测试其他服务可用性时暂时关闭
        MAX_POP_OUT_DISTANCE_RATIO = 2.5  # 如果起点在障碍物中，将其弹出，此处规定弹出距离最大值，为相对于车辆半径的比例

        # * 高程图的前处理
        FILTER_RADIUS = 2  # 引导滤波，窗口半径
        FILTER_EPS = 1e-3  # 引导滤波，正则化参数
        FILTER_SWITH_ON = False  # 是否启用滤波

        # ==============================================================================
        # 损失计算的相关参数
        # ==============================================================================
        # * 这四个是基于路径形态的惩罚，影响rs路径的评估与累计代价的计算
        SB_PENALTY = 100.0  # 换向惩罚（非系数）
        BACK_PENALTY = 5.0  # 倒车惩罚系数（仅用于rs路径评估，不参与累计代价计算）
        STEER_CHANGE_PENALTY = 5.0  # 转向变化惩罚系数（弧度角基础上的系数）
        STEER_PENALTY = 1.0  # 转向惩罚系数（弧度角基础上的系数）

        # * H_WEIGHT是影响比较大的一个系数，在他较大时会更倾向于直奔接近终点的地方
        # * 但是路径可能是次优的，也有可能导致规划失败
        H_WEIGHT = 1.0  # 启发项权重，节点代价为: 已知路径代价 + 启发式代价 + 坡度惩罚 + 崎岖度惩罚
        SLOPE_WEIGHT = 10  # 惩罚系数：格点的坡度(tan值)
        ROUGH_WEIGHT = 30  # 惩罚系数：格点的崎岖度(窗口内标准差)

        # ==============================================================================
        # 解析扩张相关
        # ==============================================================================
        # * 启发项物理含义：预期到达的距离，单位 [m]
        H_HIGH = 5.0  # 此处规定两个启发项阈值，频率在他们之间时线性递减
        H_LOW = 0.5

        # * N为解析扩张频率(即迭代搜索轮数)，此处规定上下界
        N_MAX = 10
        N_MIN = 1

        # * 在验证解析扩张得到的路径是否碰撞时，先拒绝掉代价过高的路径提高验证效率
        # 某一次循环中的cost列表 [20.00284761554539, 179.13205454838817, 180.59115325247083, ...]
        RS_COST_REJECTION_RATIO = 2  # 相对于最低代价的比例，超出的路径会被直接拒绝不进行验证

        # ==============================================================================
        # 路径平滑相关
        # ! 最终没有使用路径平滑
        # ==============================================================================
        LEARN_RATE = 0.1
        ITERATIONS: int = 100  # 迭代次数
        WEIGHT_SMOOTH = 20
        WEIGHT_OBSTACLE = 1.0
        WEIGHT_FIDELITY = 0.2
        OBSTACLE_SIGMA = 1.0

    class Explore:
        # 以起点与终点之间距离为基准，按半径分割为不同区域
        # 如果最高不是1，在不同的地图尺度下表现应该会有差异
        R_RATIO_LIST = [0.0, 0.3, 0.6, 0.8, 1.0]
        # 更希望在距离终点更近的区域内采样，此处对面积惩罚，各区域采样点数正比于惩罚后面积
        # 需要比上面的RATIO_LIST的长度短1，如果过短会直接用最小值填充
        OUT_AREA_PANELTYS = [1.0, 0.5, 0.25, 0.05]
        # 期望的总采样点数
        SUM_SAMPLE_NUM = 20
        # 计算不考虑障碍的rs路径长度作为代价
        # 由已知段（起点到候选点）与未知段（候选点到终点）组成，此处对未知段长度惩罚
        UNKOWN_PATH_PANELTY = 2
        # 采样数量相对于期望采样数的比例（因为随机采样后要根据障碍物拒绝，所以要大一些）
        SAMPLER_RATIO = 50

    class Debug:
        use_profile = True
        use_profile = False
        test_sim_origin = False

    C = Car()
    A = AStar()
    E = Explore()


S = Settings()
C = S.C
A = S.A
E = S.E

# 确保不同 import 路径指向同一模块对象
aliases = [
    "utils",
    "path_planner.utils",
]

for alias in aliases:
    if alias not in sys.modules:
        sys.modules[alias] = sys.modules[__name__]


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
        self._x = x
        self._y = y
        if deg:
            self._yaw = math.radians(yaw)
        else:
            self._yaw = yaw
        pi = math.pi
        self._yaw = (self._yaw + pi) % (2 * pi) - pi

    def __sub__(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __xor__(self, other: "Pose2D") -> PoseDiff:
        distance = self - other
        diff_abs = abs(self.yaw_deg360 - other.yaw_deg360)
        yaw_diff = min(diff_abs, 360 - diff_abs)
        return PoseDiff(dist=distance, yaw_diff_deg=yaw_diff, yaw_diff_rad=yaw_diff * math.pi / 180)

    @cached_property
    def t(self) -> NDArray[np.float64]:
        return np.array([[self.x], [self.y]])

    @cached_property
    def SO2(self) -> NDArray[np.float64]:
        "表示在全局坐标系"
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        return np.array([[cos_y, -sin_y], [sin_y, cos_y]])

    @cached_property
    def SO2inv(self) -> NDArray[np.float64]:
        "表示在本体坐标系"
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        return np.array([[cos_y, sin_y], [-sin_y, cos_y]])

    @cached_property
    def SE2(self) -> NDArray[np.float64]:
        "表示在全局坐标系"
        c = math.cos(self._yaw)
        s = math.sin(self._yaw)
        return np.array([[c, -s, self.x], [s, c, self.y], [0, 0, 1]])

    @cached_property
    def SE2inv(self) -> NDArray[np.float64]:
        "表示在本体坐标系"
        c = math.cos(self._yaw)
        s = math.sin(self._yaw)
        return np.array(
            [
                [c, s, -c * self.x - s * self.y],
                [-s, c, s * self.x - c * self.y],
                [0, 0, 1],
            ]
        )

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @cached_property
    def xy(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y])

    @property
    def yaw_rad(self) -> float:
        "偏航角，弧度制"
        return self._yaw

    @cached_property
    def yaw_deg180(self) -> float:
        """返回 [-180, 180) 的角度值"""
        deg = math.degrees(self._yaw) % 360
        return (deg + 180) % 360 - 180

    @cached_property
    def yaw_deg360(self) -> float:
        """返回 [0, 360) 的角度值"""
        return math.degrees(self._yaw) % 360

    def __str__(self):
        return f"{self.x}, {self.y}, {self.yaw_deg360}"

    def __repr__(self):
        return f"{self.x}, {self.y}, {self.yaw_deg360}"
