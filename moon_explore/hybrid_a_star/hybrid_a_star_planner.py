"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)
and me

"""

import heapq
import math
from dataclasses import dataclass
from math import cos, sin, tan
from pathlib import Path
from pathlib import Path as fPath
from typing import Dict, Generator, List, Set, Tuple

import numpy as np
import reeds_shepp_path_planning as rs
from dynamic_programming_heuristic import ANode, ANodeProto, calc_distance_heuristic
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from utils import Pose2D, Settings, angle_mod

S = Settings()
C = S.C
A = S.A


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
    cost: float = 0  # TODO: 注意一下goal的cost

    def __repr__(self):
        return f"Node({self.x_index},{self.y_index},{self.yaw_index})"


class HPath:
    def __init__(
        self, x_list: List[float], y_list: List[float], yaw_list: List[float], direction_list: List[bool], cost: float
    ):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost


class HMap:
    """
    关于地图的索引方式：

    使用numpy二维数组存储地图，考虑一个正常的矩阵写法，
    右上角[0,0]定为原点，x轴指向右方，y轴指向下方，
    因此在提取某一坐标处地图信息时，需要使用map[y,x]的形式进行索引
    """

    def __init__(
        self,
        ob_map: NDArray[np.bool_],
        resolution: float = Settings.AStar.XY_GRID_RESOLUTION,
        yaw_resolution: float = A.YAW_GRID_RESOLUTION,
        rr: float = Settings.Car.BUBBLE_R,
    ) -> None:
        """
        Args:
            ob_map (NDArray[np.bool_]): 二维数组，True表示有障碍物
            resolution (float, optional): 地图分辨率 [m]
            yaw_resolution (float): 偏航角的分辨率 [rad]
            rr (float, optional): 巡视器安全班级 [m]
        """
        self.obstacle_map = ob_map
        self.resolution = resolution
        self.yaw_resolution = yaw_resolution
        self.rr = rr

        # 地图参数
        self.max_x_index, self.max_y_index = self.obstacle_map.shape
        self.min_x_index, self.min_y_index = 0, 0
        self.x_width, self.y_width = self.obstacle_map.shape
        self.min_yaw_index = round(-math.pi / yaw_resolution) - 1
        self.max_yaw_index = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw_index - self.min_yaw_index)

        # 根据半径膨胀
        distance_map: NDArray[np.float64] = distance_transform_edt(~ob_map)  # type: ignore
        self.euclidean_dilated_ob_map = distance_map <= rr / resolution * 1.2  # TODO: 安全系数

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
        "计算扁平化的索引，只考虑xy"
        return (node.y_index - self.min_y_index) * self.x_width + (node.x_index - self.min_x_index)

    def world_to_map(self, x: float, y: float, yaw: float) -> Tuple[int, int, int]:
        return round(x / self.resolution), round(y / self.resolution), round(yaw / self.yaw_resolution)

    def world_to_map_2d(self, x: float, y: float) -> Tuple[int, int]:
        return round(x / self.resolution), round(y / self.resolution)

    def build_kdtree(self) -> cKDTree:
        """
        从 obstacle_map 中提取障碍物坐标，并构建 KDTree
        """
        obstacle_indices = np.argwhere(self.obstacle_map)  # shape (N, 2)
        self.kd_tree_points = obstacle_indices * self.resolution  # 将地图索引转换为世界坐标（米）
        self.kd_tree_points = self.kd_tree_points[:, [1, 0]]  # 地图的索引是yx顺序的，进行交换
        return cKDTree(self.kd_tree_points)


class HybridAStarPlanner:
    def __init__(self, map: HMap) -> None:
        self.map = map
        self.obstacle_kd_tree = map.build_kdtree()
        self.kd_tree_points = self.map.kd_tree_points

    def planning(self, start: Pose2D, goal: Pose2D):
        """
        start: start pose
        goal: goal pose
        """

        start_node = HNode(
            *self.map.world_to_map(start.x, start.y, start.yaw_rad),
            True,
            [start.x],
            [start.y],
            [start.yaw_rad],
            [True],
            cost=0,
        )
        goal_node = HNode(
            *self.map.world_to_map(goal.x, goal.y, goal.yaw_rad),
            True,
            [goal.x],
            [goal.y],
            [goal.yaw_rad],
            [True],
        )

        openList: Dict[int, HNode] = {}
        closedList: Dict[int, HNode] = {}

        h_dp = calc_distance_heuristic(self.map, goal_node.x_list[-1], goal_node.y_list[-1])

        pq: List[Tuple[float, int]] = []
        openList[self.map.calc_index(start_node)] = start_node
        heapq.heappush(pq, (self.calc_cost(start_node, h_dp), self.map.calc_index(start_node)))
        final_path = None

        while True:
            if not openList:
                print("Error: Cannot find path, No open set")
                return HPath([], [], [], [], 0)

            cost, c_id = heapq.heappop(pq)
            if c_id in openList:
                current = openList.pop(c_id)
                closedList[c_id] = current
            else:
                continue

            final_path = self.update_node_with_analytic_expansion(current, goal_node)

            if final_path is not None:
                print("path found")
                break

            new_push_neighbor: Set[int] = set()
            for neighbor in self.get_neighbors(current):
                neighbor_index = self.map.calc_index(neighbor)
                if neighbor_index in closedList:
                    continue
                if neighbor_index not in openList or openList[neighbor_index].cost > neighbor.cost:
                    new_push_neighbor.add(neighbor_index)
                    openList[neighbor_index] = neighbor

            for neighbor_index in new_push_neighbor:
                heapq.heappush(pq, (self.calc_cost(openList[neighbor_index], h_dp), neighbor_index))

        path = self.get_final_path(closedList, final_path)
        return path

    def calc_cost(self, n: HNode, h_dp: Dict[int, ANode]):
        ind = self.map.calc_index_2d(n)
        if ind not in h_dp:
            return n.cost + 999999999  # d collision cost
        return n.cost + A.H_COST * h_dp[ind].cost

    def check_car_collision(self, x_list: List[float], y_list: List[float], yaw_list: List[float]):
        for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
            cx = i_x + C.BUBBLE_DIST * cos(i_yaw)
            cy = i_y + C.BUBBLE_DIST * sin(i_yaw)

            ids = self.obstacle_kd_tree.query_ball_point([cx, cy], C.BUBBLE_R, p=2)
            near_ob_points = self.map.kd_tree_points[ids, :]

            if not ids:
                continue

            # 写着i实际上并不是索引值
            if not self.rectangle_check(i_x, i_y, i_yaw, near_ob_points):
                return False  # collision

        return True  # no collision

    def rectangle_check(self, x: float, y: float, yaw: float, ob_points: NDArray[np.float64]) -> bool:
        # 传进来的点数组为(N,2)，转换为齐次坐标(3,N)
        rot = Pose2D(x, y, yaw).SE2inv
        ob_points = ob_points.T
        ones = np.ones((1, ob_points.shape[1]))
        ob_points_homo = np.vstack([ob_points, ones])  # (3, N)
        transformed = rot @ ob_points_homo  # (3, N)
        ox, oy = transformed[0], transformed[1]
        for rx, ry in zip(ox, oy):

            if not (rx > C.LF or rx < -C.LB or ry > C.W / 2.0 or ry < -C.W / 2.0):
                return False  # collision

        return True  # no collision

    def analytic_expansion(self, current: HNode, goal: HNode):
        start_x = current.x_list[-1]
        start_y = current.y_list[-1]
        start_yaw = current.yaw_list[-1]

        goal_x = goal.x_list[-1]
        goal_y = goal.y_list[-1]
        goal_yaw = goal.yaw_list[-1]

        max_curvature = math.tan(C.MAX_STEER) / C.WB
        paths: List[rs.RPath] = rs.calc_paths(
            Pose2D(start_x, start_y, start_yaw),
            Pose2D(goal_x, goal_y, goal_yaw),
            max_curvature,
            step_size=A.MOTION_RESOLUTION,
        )

        if not paths:
            return None

        best_path, best = None, None

        for path in paths:
            if self.check_car_collision(path.x, path.y, path.yaw):
                cost = self.calc_rs_path_cost(path)
                if not best or best > cost:
                    best = cost
                    best_path = path

        return best_path

    def update_node_with_analytic_expansion(self, current: HNode, goal: HNode):
        path = self.analytic_expansion(current, goal)

        if path:
            f_x = path.x[1:]
            f_y = path.y[1:]
            f_yaw = path.yaw[1:]

            f_cost = current.cost + self.calc_rs_path_cost(path)
            f_parent_index = self.map.calc_index(current)

            fd: List[bool] = []
            for d in path.directions[1:]:
                fd.append(d >= 0)

            f_steer = 0.0
            f_path = HNode(
                current.x_index,
                current.y_index,
                current.yaw_index,
                current.direction,
                f_x,
                f_y,
                f_yaw,
                fd,
                cost=f_cost,
                parent_index=f_parent_index,
                steer=f_steer,
            )
            return f_path

        return None

    def calc_rs_path_cost(self, reed_shepp_path: rs.RPath):
        cost = 0.0
        for length in reed_shepp_path.lengths:
            if length >= 0:  # forward
                cost += length
            else:  # back
                cost += abs(length) * A.BACK_PENALTY

        # switch back penalty
        for i in range(len(reed_shepp_path.lengths) - 1):
            # switch back
            if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
                cost += A.SB_PENALTY

        # steer penalty
        for course_type in reed_shepp_path.ctypes:
            if course_type != "S":  # curve
                cost += A.STEER_PENALTY * abs(C.MAX_STEER)

        # ==steer change penalty
        # calc steer profile
        n_ctypes = len(reed_shepp_path.ctypes)
        u_list = [0.0] * n_ctypes
        for i in range(n_ctypes):
            if reed_shepp_path.ctypes[i] == "R":
                u_list[i] = -C.MAX_STEER
            elif reed_shepp_path.ctypes[i] == "L":
                u_list[i] = C.MAX_STEER

        for i in range(len(reed_shepp_path.ctypes) - 1):
            cost += A.STEER_CHANGE_PENALTY * abs(u_list[i + 1] - u_list[i])

        return cost

    def calc_motion_inputs(self) -> Generator[Tuple[float, int], None, None]:
        for steer in np.concatenate((np.linspace(-C.MAX_STEER, C.MAX_STEER, A.N_STEER), [0.0])):
            for d in [1, -1]:
                yield (steer, d)

    def get_neighbors(self, current: HNode) -> Generator[HNode, None, None]:
        for steer, d in self.calc_motion_inputs():
            node = self.calc_next_node(current, steer, d)
            if node and self.map.verify_index(node):
                yield node

    def move(self, x: float, y: float, yaw: float, distance: float, steer: float, L: float = C.WB):
        x += distance * cos(yaw)
        y += distance * sin(yaw)
        yaw = angle_mod(yaw + distance * tan(steer) / L)  # distance/2

        return x, y, yaw

    def calc_next_node(self, current: HNode, steer: float, direction: int):
        x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

        arc_l = A.XY_GRID_RESOLUTION * 1.5
        x_list, y_list, yaw_list, direction_list = [], [], [], []
        x_list: List[float]
        y_list: List[float]
        yaw_list: List[float]
        direction_list: List[bool]
        for _ in np.arange(0, arc_l, A.MOTION_RESOLUTION):
            x, y, yaw = self.move(x, y, yaw, A.MOTION_RESOLUTION * direction, steer)
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)
            direction_list.append(direction == 1)

        if not self.check_car_collision(x_list, y_list, yaw_list):
            return None

        d = direction == 1
        added_cost: float = 0.0

        if d != current.direction:
            added_cost += A.SB_PENALTY

        # steer penalty
        added_cost += A.STEER_PENALTY * abs(steer)

        # steer change penalty
        added_cost += A.STEER_CHANGE_PENALTY * abs(current.steer - steer)

        cost = current.cost + added_cost + arc_l

        node = HNode(
            *self.map.world_to_map(x, y, yaw),
            d,
            x_list,
            y_list,
            yaw_list,
            direction_list,
            parent_index=self.map.calc_index(current),
            cost=cost,
            steer=steer,
        )

        return node

    def get_final_path(self, closed: Dict[int, HNode], goal_node: HNode):
        reversed_x, reversed_y, reversed_yaw = (
            list(reversed(goal_node.x_list)),
            list(reversed(goal_node.y_list)),
            list(reversed(goal_node.yaw_list)),
        )
        direction = list(reversed(goal_node.directions))
        nid = goal_node.parent_index
        final_cost = goal_node.cost

        while nid:
            n = closed[nid]
            reversed_x.extend(list(reversed(n.x_list)))
            reversed_y.extend(list(reversed(n.y_list)))
            reversed_yaw.extend(list(reversed(n.yaw_list)))
            direction.extend(list(reversed(n.directions)))

            nid = n.parent_index

        reversed_x = list(reversed(reversed_x))
        reversed_y = list(reversed(reversed_y))
        reversed_yaw = list(reversed(reversed_yaw))
        direction = list(reversed(direction))

        # adjust first direction
        direction[0] = direction[1]

        path = HPath(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

        return path


def main():
    import matplotlib.pyplot as plt

    from plot.plot_utils import MAP_PASSABLE, plot_binary_map, plt_tight_show

    print("Start Hybrid A* planning")
    map_npy = Path(__file__).resolve().parents[2] / "resource/map_passable.npy"

    # Set Initial parameters
    start = Pose2D(40.0, 10.0, 90.0, deg=True)
    goal = Pose2D(45.0, 32, 90.0, deg=True)

    print("start : ", start)
    print("goal : ", goal)

    map = HMap.from_file(map_npy)
    planner = HybridAStarPlanner(map)
    path = planner.planning(start, goal)

    x = [x * 10 for x in path.x_list]
    y = [x * 10 for x in path.y_list]
    if show_animation:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plot_binary_map(MAP_PASSABLE, ax)
        plt_tight_show()

    print(__file__ + " done!!")


if __name__ == "__main__":
    import datetime
    import inspect
    import sys
    from pathlib import Path as fPath

    from line_profiler import LineProfiler

    use_profile = True
    # use_profile = False
    if use_profile:
        show_animation = False
        lp = LineProfiler()
        current_module = sys.modules[__name__]

        # 获取所有用户定义的函数（跳过内置、导入的）
        for name, obj in inspect.getmembers(current_module):
            if inspect.isfunction(obj):
                lp.add_function(obj)  # 普通函数
            elif inspect.isclass(obj):
                for meth_name, meth in inspect.getmembers(obj, inspect.isfunction):
                    lp.add_function(meth)  # 类方法

        lp_wrapper = lp(main)
        lp_wrapper()
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        name = fPath(__file__).stem
        short_name = "_".join(name.split("_")[:2])  # 取前两个单词组合
        profile_filename = f"profile_{short_name}_{timestamp}.txt"
        with open(profile_filename, "w") as f:
            lp.print_stats(sort=True, stream=f)
    else:
        show_animation = True
        main()
