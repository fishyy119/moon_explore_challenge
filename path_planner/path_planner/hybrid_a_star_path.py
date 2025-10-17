import math
from typing import List

import numpy as np
from rs_planning import RPath
from utils import Pose2D


class HPath:
    def __init__(
        self, x_list: List[float], y_list: List[float], yaw_list: List[float], direction_list: List[bool], cost: float
    ):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost

    @classmethod
    def from_rpath(cls, rpath: RPath):
        # 将 directions 转为 bool: 1 -> True, -1 -> False
        direction_list = [d == 1 for d in rpath.directions]

        # 创建 HPath
        return cls(
            x_list=rpath.x.tolist(),
            y_list=rpath.y.tolist(),
            yaw_list=rpath.yaw.tolist(),
            direction_list=direction_list,
            cost=rpath.L,
        )


def generate_forward_path(start_pose: Pose2D, distance: float, step=0.01) -> HPath:
    """
    生成前进/后退的模板路径

    Args:
        start_pose (Pose2D): 起点位姿，用于对齐
        distance (float): 前进/后退距离，后退为负数
        step (float, optional): 路径步长

    Returns:
        HPath: 路径结果，和规划器输出保持一致
    """
    x0, y0, yaw0 = start_pose.x, start_pose.y, start_pose.yaw_rad
    n = int(abs(distance) / step) + 1
    forward = distance >= 0

    # 生成距离序列
    d = np.linspace(0, distance, n)  # distance 为负时自动是递减
    # 坐标计算（广播运算）
    cos_yaw = math.cos(yaw0)
    sin_yaw = math.sin(yaw0)
    x_list = x0 + d * cos_yaw
    y_list = y0 + d * sin_yaw

    # yaw 全部一样
    yaw_list = np.full_like(d, yaw0)
    dir_list = np.full_like(d, forward, dtype=np.bool_)

    return HPath(x_list.tolist(), y_list.tolist(), yaw_list.tolist(), dir_list.tolist(), cost=abs(distance))


def generate_circle_path(start_pose: Pose2D, radius: float, arc_angle: float, step=0.01) -> HPath:
    """
    生成圆弧模板路径

    Args:
        start_pose (Pose2D): 起点位姿，用于对齐
        radius (float): 圆弧半径，正值左转，负值右转
        arc_angle (float): 圆弧角度（弧度制），正值逆时针，负值顺时针
        step (float, optional): 路径步长

    Returns:
        HPath: 路径结果，和规划器输出保持一致
    """
    arc_len = abs(radius * arc_angle)
    n = int(arc_len / step) + 1
    forward = (radius * arc_angle) > 0

    # 在标准坐标系(0,0,0朝向)下生成模板
    # x-前 y-左
    theta = np.linspace(0, arc_angle, n)  # 圆心角
    cx, cy = 0.0, radius  # 圆心在前进方向左侧 radius 距离
    x_std = cx + radius * np.sin(theta)
    y_std = cy - radius * np.cos(theta)
    yaw_std = theta  # 切线方向（相对于标准起点）

    # 前进/后退判定
    dir_list = np.full_like(theta, forward, dtype=np.bool_)

    # 应用起点位姿的 SE(2) 变换
    cos_yaw = math.cos(start_pose.yaw_rad)
    sin_yaw = math.sin(start_pose.yaw_rad)
    x_new = cos_yaw * x_std - sin_yaw * y_std + start_pose.x
    y_new = sin_yaw * x_std + cos_yaw * y_std + start_pose.y
    yaw_new = yaw_std + start_pose.yaw_rad

    return HPath(x_new.tolist(), y_new.tolist(), yaw_new.tolist(), dir_list.tolist(), cost=arc_len)


def generate_figure8_path(start_pose: Pose2D, radius: float, step=0.01) -> HPath:
    """
    生成前进八字形路径（∞），复用圆弧模板

    Args:
        start_pose (Pose2D): 起点位姿
        radius (float): 八字形每半圆的半径
        step (float, optional): 路径步长

    Returns:
        HPath: 路径结果
    """
    # 生成上半圆（180°）
    circle_1 = generate_circle_path(start_pose, radius, np.copysign(np.pi * 2, radius), step)

    # 下半圆起点位姿是上半圆末点
    last_idx = -1
    end_pose = Pose2D(x=circle_1.x_list[last_idx], y=circle_1.y_list[last_idx], yaw=circle_1.yaw_list[last_idx])

    circle_2 = generate_circle_path(end_pose, -radius, np.copysign(np.pi * 2, -radius), step)

    # 拼接两段
    x_list = circle_1.x_list + circle_2.x_list
    y_list = circle_1.y_list + circle_2.y_list
    yaw_list = circle_1.yaw_list + circle_2.yaw_list
    dir_list = circle_1.direction_list + circle_2.direction_list
    cost = circle_1.cost + circle_2.cost

    return HPath(x_list, y_list, yaw_list, dir_list, cost)


def test_generate_path():
    # 生成圆弧路径
    start_pose = Pose2D(3.0, 0.0, 50.0, deg=True)
    radius = 2.0  # 正值左转
    arc_angle = -np.pi / 2
    path = generate_forward_path(start_pose, -3.4, step=0.05)
    # path = generate_circle_path(start_pose, radius, arc_angle, step=0.05)
    # path = generate_figure8_path(start_pose, radius, step=0.05)
    print(path.direction_list[0])

    # 绘制轨迹
    plt.figure(figsize=(6, 6))
    plt.plot(path.x_list, path.y_list, "b-", label="Circle Path")
    plt.plot(start_pose.x, start_pose.y, "go", label="Start")
    plt.quiver(
        path.x_list[::5],
        path.y_list[::5],
        np.cos(path.yaw_list[::5]),
        np.sin(path.yaw_list[::5]),
        color="r",
        scale=10,
        width=0.005,
        label="Yaw",
    )
    plt.axis("equal")
    plt.grid(True)
    # plt.legend()
    plt.title("Circle Path Test")
    plt.show()


# 调用测试
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_generate_path()
