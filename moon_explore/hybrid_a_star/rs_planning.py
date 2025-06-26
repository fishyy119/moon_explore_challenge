"""

Reeds Shepp path planner sample code

author Atsushi Sakai(@Atsushi_twi)
co-author Videh Patel(@videh25) : Added the missing RS paths
and me

"""

import math
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from moon_explore.hybrid_a_star.rs_patterns import path_funcs
from utils import Pose2D, plot_arrow


class RPath:
    def __init__(self):
        self.lengths: List[float] = []  # course segment length  (negative value is backward segment)
        self.ctypes: List[str] = []  # course segment type char ("S": straight, "L": left, "R": right)
        self.L: float = 0.0  # Total lengths of the path
        self.x: NDArray[np.float64] = np.empty(1)  # x positions
        self.y: NDArray[np.float64] = np.empty(1)  # y positions
        self.yaw: NDArray[np.float64] = np.empty(1)  # orientations [rad]
        self.directions: List[int] = []  # directions (1:forward, -1:backward)


def set_path(paths: List[RPath], lengths: List[float], ctypes: List[str], step_size: float):
    """
    这东西难道实际上不是add吗？
    """
    path = RPath()
    path.ctypes = ctypes
    path.lengths = lengths
    path.L = sum(np.abs(lengths))

    # check same path exist
    for i_path in paths:
        type_is_same = i_path.ctypes == path.ctypes
        length_is_close = (sum(np.abs(i_path.lengths)) - path.L) <= step_size
        if type_is_same and length_is_close:
            return paths  # same path found, so do not insert path

    # check path is long enough
    if path.L <= step_size:
        return paths  # too short, so do not insert path

    paths.append(path)
    return paths


_REFLECT_MAP = {"L": "R", "R": "L", "S": "S"}


def reflect(steering_directions: List[str]) -> List[str]:
    return [_REFLECT_MAP[dirn] for dirn in steering_directions]


def timeflip(travel_distances: List[float]) -> List[float]:
    return [-x for x in travel_distances]


def generate_path(
    p0: Pose2D,
    p1: Pose2D,
    max_curvature: float,
    step_size: float,
) -> List[RPath]:
    # 归一化
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    dth = p1.yaw_rad - p0.yaw_rad
    c = math.cos(p0.yaw_rad)
    s = math.sin(p0.yaw_rad)
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    step_size *= max_curvature

    paths: List[RPath] = []

    for path_func in path_funcs.values():
        flag, travel_distances, steering_dirns = path_func(x, y, dth)
        if flag:
            for distance in travel_distances:
                if 0.1 * sum([abs(d) for d in travel_distances]) < abs(distance) < step_size:
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, y, -dth)
        if flag:
            for distance in travel_distances:
                if 0.1 * sum([abs(d) for d in travel_distances]) < abs(distance) < step_size:
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(x, -y, -dth)
        if flag:
            for distance in travel_distances:
                if 0.1 * sum([abs(d) for d in travel_distances]) < abs(distance) < step_size:
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, -y, dth)
        if flag:
            for distance in travel_distances:
                if 0.1 * sum([abs(d) for d in travel_distances]) < abs(distance) < step_size:
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

    return paths


def calc_interpolate_dists_list(lengths: List[float], step_size: float) -> List[NDArray[np.float64]]:
    interpolate_dists_list: List[NDArray[np.float64]] = []
    for length in lengths:
        d_dist = step_size if length >= 0.0 else -step_size

        interp_core = np.arange(0.0, length, d_dist, dtype=np.float64)
        interp_dists = np.empty(len(interp_core) + 1, dtype=np.float64)
        interp_dists[:-1] = interp_core
        interp_dists[-1] = length

        interpolate_dists_list.append(interp_dists)

    return interpolate_dists_list


def generate_local_course(
    lengths: List[float],
    modes: List[str],
    max_curvature: float,
    step_size: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    interpolate_dists_list = calc_interpolate_dists_list(lengths, step_size * max_curvature)
    total_len = sum(len(arr) for arr in interpolate_dists_list)
    xs = np.empty(total_len, dtype=np.float64)
    ys = np.empty_like(xs)
    yaws = np.empty_like(xs)
    directions = np.empty_like(xs, dtype=np.int32)

    origin_x, origin_y, origin_yaw = 0.0, 0.0, 0.0
    idx = 0

    for interp_dists, mode, length in zip(interpolate_dists_list, modes, lengths):
        n = len(interp_dists)
        x_arr, y_arr, yaw_arr, dir_arr = interpolate_vectorized(
            interp_dists, length, mode, max_curvature, origin_x, origin_y, origin_yaw
        )
        xs[idx : idx + n] = x_arr
        ys[idx : idx + n] = y_arr
        yaws[idx : idx + n] = yaw_arr
        directions[idx : idx + n] = dir_arr

        origin_x = x_arr[-1]
        origin_y = y_arr[-1]
        origin_yaw = yaw_arr[-1]
        idx += n

    return xs, ys, yaws, directions


def interpolate_vectorized(
    dists: NDArray[np.float64],
    length: float,
    mode: str,
    max_curvature: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    if mode == "S":
        # 直线段，x,y,yaw都可批量计算
        x = origin_x + dists / max_curvature * math.cos(origin_yaw)
        y = origin_y + dists / max_curvature * math.sin(origin_yaw)
        yaw = np.full_like(dists, origin_yaw)
    else:
        # 曲线段
        ldx = np.sin(dists) / max_curvature
        ldy = np.zeros_like(dists)
        yaw = None
        if mode == "L":
            ldy = (1.0 - np.cos(dists)) / max_curvature
            yaw = origin_yaw + dists
        else:  # elif mode == "R":
            ldy = (1.0 - np.cos(dists)) / (-max_curvature)
            yaw = origin_yaw - dists

        cos_oy = math.cos(-origin_yaw)
        sin_oy = math.sin(-origin_yaw)
        gdx = cos_oy * ldx + sin_oy * ldy
        gdy = -sin_oy * ldx + cos_oy * ldy
        x = origin_x + gdx
        y = origin_y + gdy

    # 方向：标量，由长度正负决定
    direction = 1 if length > 0 else -1
    return x, y, yaw, np.full_like(dists, direction, dtype=int)


def calc_paths(
    p0: Pose2D,
    p1: Pose2D,
    maxc: float,
    step_size: float,
) -> List[RPath]:
    """
    计算从起点到终点的所有可行 Reeds-Shepp 路径，并转换为全局坐标系下的路径。

    本函数在单位曲率空间内生成一组候选路径，随后将其缩放并旋转平移回实际坐标系。

    Args:
        p0 (Pose2D): 起始位姿，包含坐标 (x, y) 和朝向角 yaw。
        p1 (Pose2D): 目标位姿，格式同上。
        maxc (float): 转弯的最大曲率。
        step_size (float): 路径采样的步长，决定生成路径的分辨率，单位 m。

    Returns:
        List[RPath]: 包含所有可行路径的列表，每个路径使用 RPath 类描述。
    """
    paths = generate_path(p0, p1, maxc, step_size)
    for path in paths:
        xs, ys, yaws, directions = generate_local_course(path.lengths, path.ctypes, maxc, step_size)

        # 归一化的反向变换
        local_pts = np.vstack([xs, ys, np.ones_like(xs)])  # shape: [3, N]
        global_pts = p0.SE2 @ local_pts  # shape: [3, N]

        path.x = global_pts[0, :]
        path.y = global_pts[1, :]

        path.yaw = (yaws + p0.yaw_rad + np.pi) % (2 * np.pi) - np.pi

        # 其他字段原样更新
        path.directions = directions.tolist()
        path.lengths = [l / maxc for l in path.lengths]
        path.L /= maxc

    return paths


def reeds_shepp_path_planning(
    sx: float,
    sy: float,
    syaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    maxc: float,
    step_size: float = 0.2,
) -> RPath | None:
    p0 = Pose2D(sx, sy, syaw)
    p1 = Pose2D(gx, gy, gyaw)
    paths = calc_paths(p0, p1, maxc, step_size)
    if not paths:
        return None  # could not generate any path
    best_path = min(paths, key=lambda p: abs(p.L))

    return best_path


def main():

    print("Reeds Shepp path planner sample start!!")

    start_x = -1.0  # [m]
    start_y = -4.0  # [m]
    start_yaw = np.deg2rad(-20.0)  # [rad]

    end_x = 5.0  # [m]
    end_y = 5.0  # [m]
    end_yaw = np.deg2rad(25.0)  # [rad]

    curvature = 0.1
    step_size = 0.05

    result_path = reeds_shepp_path_planning(start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature, step_size)

    if result_path is None:
        assert False, "No path"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(result_path.x, result_path.y, label="final course " + str(result_path.ctypes))
        print(f"{result_path.lengths=}")

        # plotting
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)

        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    import datetime
    import inspect
    import sys
    from pathlib import Path as fPath

    import matplotlib.pyplot as plt
    from line_profiler import LineProfiler

    use_profile = True
    use_profile = False
    if use_profile:
        show_animation = False
        lp = LineProfiler()
        current_module = sys.modules[__name__]

        # 获取所有用户定义的函数（跳过内置、导入的）
        for name, obj in inspect.getmembers(current_module, inspect.isfunction):
            lp.add_function(obj)
        lp_wrapper = lp(main)
        lp_wrapper()
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        name = fPath(__file__).stem
        short_name = "_".join(name.split("_")[:2])  # 取前两个单词组合
        profile_filename = f"profile_{short_name}_{timestamp}.txt"
        with open(profile_filename, "w") as f:
            lp.print_stats(stream=f)
    else:
        show_animation = True
        main()
