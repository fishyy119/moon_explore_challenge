"""

Reeds Shepp path planner sample code

author: Atsushi Sakai(@Atsushi_twi)
co-author: Videh Patel(@videh25) : Added the missing RS paths
and: me

version: 2025-06-27 14:00
"""

import math
from functools import wraps
from typing import Callable, Dict, List, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray


class PathFunctionResult(NamedTuple):
    success: bool
    distances: List[float]
    directions: List[str]


PathFunction = Callable[[float, float, float], PathFunctionResult]
RawPathFunction = Callable[[float, float, float], Tuple[bool, List[float], List[str]]]
path_funcs: Dict[str, PathFunction] = {}


# 在原始实现基础上的一层封装
def wrap_as_result(fn: RawPathFunction) -> PathFunction:
    @wraps(fn)
    def wrapper(x: float, y: float, phi: float) -> PathFunctionResult:
        success, params, path = fn(x, y, phi)
        return PathFunctionResult(success, params, path)

    return wrapper


# 用于注册各个基本函数到`path_funcs`
def register(name: str):
    def decorator(fn: RawPathFunction):
        if name in path_funcs:
            raise ValueError(f"[register] Duplicate registration: '{name}' already exists.")
        path_funcs[name] = wrap_as_result(fn)
        return path_funcs[name]

    return decorator


def polar(x, y):
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


def mod2pi(x: float) -> float:
    # Be consistent with fmod in cplusplus here.
    v = np.mod(x, np.copysign(2.0 * math.pi, x))
    if v < -math.pi:
        v += 2.0 * math.pi
    else:
        if v > math.pi:
            v -= 2.0 * math.pi
    return v


@register("CSC1")
def left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ["L", "S", "L"]

    return False, [], []


@register("CSC2")
def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1**2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)

        if (t >= 0.0) and (v >= 0.0):
            return True, [t, u, v], ["L", "S", "R"]

    return False, [], []


@register("CCC1")
def left_x_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi / 2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(phi - t - u)
        return True, [t, -u, v], ["L", "R", "L"]

    return False, [], []


@register("CCC2")
def left_x_right_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi / 2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(-phi + t + u)
        return True, [t, -u, -v], ["L", "R", "L"]

    return False, [], []


@register("CCC3")
def left_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        u = math.acos(1 - u1**2 * 0.125)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(-A + theta + math.pi / 2)
        v = mod2pi(t - u - phi)
        return True, [t, u, -v], ["L", "R", "L"]

    return False, [], []


@register("CCCC1")
def left_right_x_left_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    # Solutions refering to (2 < u1 <= 4) are considered sub-optimal in paper
    # Solutions do not exist for u1 > 4
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + A + math.pi / 2)
        u = mod2pi(A)
        v = mod2pi(phi - t + 2 * u)
        if (t >= 0) and (u >= 0) and (v >= 0):
            return True, [t, u, -u, -v], ["L", "R", "L", "R"]

    return False, [], []


@register("CCCC2")
def left_x_right_left_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1**2) / 16

    if 0 <= u2 <= 1:
        u = math.acos(u2)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -u, -u, v], ["L", "R", "L", "R"]

    return False, [], []


@register("CSCC1")
def left_straight_right90_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(math.sqrt(u1**2 - 4), 2)
        t = mod2pi(theta - A + math.pi / 2)
        v = mod2pi(t - phi - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi / 2, -v], ["L", "S", "R", "L"]

    return False, [], []


@register("CSCC2")
def left_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi / 2, -v], ["L", "S", "L", "R"]

    return False, [], []


@register("CCSC1")
def left_x_right90_straight_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi + math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -v], ["L", "R", "S", "L"]

    return False, [], []


@register("CCSC2")
def left_x_right90_straight_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta + math.pi / 2)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -v], ["L", "R", "S", "R"]

    return False, [], []


@register("CCSCC")
def left_x_right90_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 4.0:
        u = math.sqrt(u1**2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -math.pi / 2, v], ["L", "R", "S", "L", "R"]

    return False, [], []


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

    @property
    def SE2(self) -> NDArray[np.float64]:
        "表示在全局坐标系"
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        return np.array([[cos_y, -sin_y, self.x], [sin_y, cos_y, self.y], [0, 0, 1]])

    @property
    def yaw_rad(self) -> float:
        "偏航角，弧度制"
        return self._yaw


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if isinstance(x, list):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(
            x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width
        )
        plt.plot(x, y)


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
    # print("Reeds Shepp path planner sample start!!")

    # ============================ 输入参数 =================================
    start_x = -1.0  # [m]
    start_y = -4.0  # [m]
    start_yaw = np.deg2rad(-20.0)  # [rad]

    end_x = 5.0  # [m]
    end_y = 5.0  # [m]
    end_yaw = np.deg2rad(25.0)  # [rad]

    curvature = 0.1  # 最大曲率
    step_size = 0.05  # 路径采样的步长 [m]
    # =====================================================================

    result_path = reeds_shepp_path_planning(start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature, step_size)

    if result_path is None:
        assert False, "No path"

    # ======================= 最终的路径结果 ==================================
    x = result_path.x.tolist()
    y = result_path.y.tolist()
    yaw = result_path.yaw.tolist()
    print(f"{x=}", end="\n\n")
    print(f"{y=}", end="\n\n")
    print(f"{yaw=}", end="\n\n")
    print(f"{result_path.directions=}", end="\n\n")  # 前进/后退
    # ======================================================================

    show_animation = True
    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(result_path.x, result_path.y, label="final course " + str(result_path.ctypes))
        # print(f"{result_path.lengths=}")

        # plotting
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)

        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
