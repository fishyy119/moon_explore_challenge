"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)
and: me

"""

import heapq
import math
from dataclasses import dataclass
from math import atan2, cos, pi, sin, tan
from pathlib import Path as fPath
from typing import Dict, Generator, List, Set, Tuple, cast

import numpy as np
import rs_planning as rs
from dynamic_programming_heuristic import ANodeProto, calc_distance_heuristic
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from utils import A, C, Pose2D, S

from path_planner import RESOURCE_DIR


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
            rr (float, optional): 巡视器安全班级 [m]
            origin (float, optional): 接收到的地图中00栅格相对于地图坐标系的位姿
        """
        self.obstacle_map = ob_map
        self.resolution = resolution
        self.yaw_resolution = yaw_resolution
        self.rr = rr
        self.edf_map: NDArray[np.float64] = distance_transform_edt(~ob_map) * resolution  # [m] # type: ignore
        self.euclidean_dilated_ob_map = self.edf_map <= rr * A.SAFETY_MARGIN_RATIO  # 根据半径膨胀

        # 地图参数
        self.max_x_index, self.max_y_index = self.obstacle_map.shape
        self.min_x_index, self.min_y_index = 0, 0
        self.x_width, self.y_width = self.obstacle_map.shape
        self.min_yaw_index = round(-pi / yaw_resolution) - 1
        self.max_yaw_index = round(pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw_index - self.min_yaw_index)

        # 输入输出要加这个转换
        self.origin_pose = origin
        self.SE2 = origin.SE2  # 内部 -> 外部
        self.SE2inv = origin.SE2inv  # 外部 -> 内部
        self.map_yaw = origin.yaw_rad  # 内部 -> 外部（加的话）

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
        obstacle_indices = np.argwhere(self.obstacle_map)  # shape (N, 2)
        self.kd_tree_points = obstacle_indices * self.resolution  # 将地图索引转换为世界坐标（米）
        self.kd_tree_points = self.kd_tree_points[:, [1, 0]]  # 地图的索引是yx顺序的，进行交换
        return cKDTree(self.kd_tree_points)


class HybridAStarPlanner:
    def __init__(self, map: HMap) -> None:
        self.map = map
        self.obstacle_kd_tree = map.build_kdtree()
        self.kd_tree_points = self.map.kd_tree_points
        self.hrs_table: NDArray[np.float64] | None
        table_file = RESOURCE_DIR / f"rs_table_{A.MAP_MAX_SIZE}x{A.MAP_MAX_SIZE}.npy"
        try:
            self.hrs_table = np.load(table_file)
        except FileNotFoundError:
            self.hrs_table = None
            print(f"RS 启发式表{str(table_file)}未找到，将禁用 RS 启发")

    def planning(self, start: Pose2D, goal: Pose2D) -> Tuple[None | HPath, str]:
        """
        输入都在外部地图坐标系表示，需要转换
        start: 起点位姿
        goal: 终点位姿
        """
        out = np.array([[start.x, goal.x], [start.y, goal.y], [1, 1]])
        inn = self.map.SE2inv @ out
        sx = inn[0, 0]
        sy = inn[1, 0]
        syaw = (self.map.map_yaw + start.yaw_rad + pi) % (2 * pi) - pi
        gx = inn[0, 1]
        gy = inn[1, 1]
        gyaw = (self.map.map_yaw + goal.yaw_rad + pi) % (2 * pi) - pi

        sidx = self.map.world_to_map(sx, sy, syaw)
        gidx = self.map.world_to_map(gx, gy, gyaw)
        if (
            sidx[0] <= self.map.min_x_index
            or sidx[1] <= self.map.min_y_index
            or sidx[0] >= self.map.max_x_index
            or sidx[1] >= self.map.max_y_index
        ) or (
            gidx[0] <= self.map.min_x_index
            or gidx[1] <= self.map.min_y_index
            or gidx[0] >= self.map.max_x_index
            or gidx[1] >= self.map.max_y_index
        ):
            return (
                None,
                f"输入坐标不在地图范围内：({start.x}, {start.y}) -> ({sx}, {sy}), ({goal.x}, {goal.y}) -> ({gx}, {gy})",
            )

        start_node = HNode(
            *sidx,
            True,
            [sx],
            [sy],
            [syaw],
            [True],
            cost=0,
        )
        goal_node = HNode(
            *gidx,
            True,
            [gx],
            [gy],
            [gyaw],
            [True],
        )
        self.map

        openList: Dict[int, HNode] = {}
        closedList: Dict[int, HNode] = {}

        self.hstar_dp = calc_distance_heuristic(self.map, goal_node.x_list[-1], goal_node.y_list[-1])
        pq: List[Tuple[float, int]] = []
        openList[self.map.calc_index(start_node)] = start_node
        heapq.heappush(
            pq,
            (
                self.calc_cost(start_node, self.calc_heuristic(start_node, goal_node)),
                self.map.calc_index(start_node),
            ),
        )
        final_path = None
        rs_cnt = 0

        while True:
            rs_cnt += 1
            if not openList:
                return None, "未搜索出有效路径"

            _, c_id = heapq.heappop(pq)
            if c_id in openList:
                current = openList.pop(c_id)
                closedList[c_id] = current
            else:
                continue

            h_total = self.calc_heuristic(current, goal_node)
            if rs_cnt >= self.calc_rs_interval(h_total):
                rs_cnt = 0
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
                heapq.heappush(pq, (self.calc_cost(openList[neighbor_index], h_total), neighbor_index))

        path = self.get_final_path(closedList, final_path)
        # s_path = self.smooth_path(path)
        # return s_path
        return path, ""

    def update_node_with_analytic_expansion(self, current: HNode, goal: HNode):
        # analytic expansion(解析扩张)
        start_x = current.x_list[-1]
        start_y = current.y_list[-1]
        start_yaw = current.yaw_list[-1]

        goal_x = goal.x_list[-1]
        goal_y = goal.y_list[-1]
        goal_yaw = goal.yaw_list[-1]

        max_curvature = tan(C.MAX_STEER) / C.WB
        paths: List[rs.RPath] = rs.calc_paths(
            Pose2D(start_x, start_y, start_yaw),
            Pose2D(goal_x, goal_y, goal_yaw),
            max_curvature,
            step_size=A.MOTION_RESOLUTION,
        )

        if not paths:
            return None

        path_costs = [self.calc_rs_path_cost(path) for path in paths]
        good_path_ids = np.argsort(path_costs)
        best_path, best_cost = None, path_costs[good_path_ids[0]]
        for idx in good_path_ids:
            idx: int
            if path_costs[idx] > best_cost * A.RS_COST_REJECTION_RATIO:
                break  # 提前退出验证
            path = paths[idx]
            if self.check_car_collision(path.x, path.y, path.yaw):
                best_cost = path_costs[idx]
                best_path = path
                break

        # update
        if best_path is not None:
            best_cost = cast(float, best_cost)
            f_x = best_path.x[1:]
            f_y = best_path.y[1:]
            f_yaw = best_path.yaw[1:]

            f_cost = current.cost + best_cost
            f_parent_index = self.map.calc_index(current)

            fd: List[bool] = []
            for d in best_path.directions[1:]:
                fd.append(d >= 0)

            f_steer = 0.0
            f_path = HNode(
                current.x_index,
                current.y_index,
                current.yaw_index,
                current.direction,
                f_x.tolist(),
                f_y.tolist(),
                f_yaw.tolist(),
                fd,
                cost=f_cost,
                parent_index=f_parent_index,
                steer=f_steer,
            )
            return f_path

        return None

    def get_neighbors(self, current: HNode) -> Generator[HNode, None, None]:
        for steer, d in self.calc_motion_inputs():
            node = self.calc_next_node(current, steer, d)
            if node and self.map.verify_index(node):
                yield node

    def calc_motion_inputs(self) -> Generator[Tuple[float, int], None, None]:
        for steer in np.concatenate((np.linspace(-C.MAX_STEER, C.MAX_STEER, A.N_STEER), [0.0])):
            for d in [1, -1]:
                yield (steer, d)

    def calc_next_node(self, current: HNode, steer: float, direction: int) -> None | HNode:
        arc_l = A.XY_GRID_RESOLUTION * 1.5
        distance = A.MOTION_RESOLUTION * direction
        steps = int(np.ceil(arc_l / A.MOTION_RESOLUTION))
        L = C.WB
        i = np.arange(0, steps + 1)  # shape=(N,)
        delta_yaw = distance * tan(steer) / L  # scalar

        yaw_array = (current.yaw_list[-1] + delta_yaw * i + np.pi) % (2 * np.pi) - np.pi
        dx = distance * np.cos(yaw_array[:-1])
        dy = distance * np.sin(yaw_array[:-1])  # 只是先更新yaw还是先更新xy的区别，此处暂时与原版本保持一致

        x_array = current.x_list[-1] + np.cumsum(dx)  # shape=(N,)
        y_array = current.y_list[-1] + np.cumsum(dy)  # shape=(N,)
        direction_array = np.full_like(x_array, direction == 1, dtype=np.bool_)

        if not self.check_car_collision(x_array, y_array, yaw_array[1:]):
            return None

        d = direction == 1
        added_cost: float = 0.0
        if d != current.direction:
            added_cost += A.SB_PENALTY  # 换档惩罚
        added_cost += A.STEER_PENALTY * abs(steer)  # 转向惩罚
        added_cost += A.STEER_CHANGE_PENALTY * abs(current.steer - steer)  # 转向改变惩罚
        cost = current.cost + added_cost + arc_l  # 额外损失 + 增加路径长度

        node = HNode(
            *self.map.world_to_map(x_array[-1], y_array[-1], yaw_array[-1]),
            d,
            x_array.tolist(),
            y_array.tolist(),
            yaw_array[1:].tolist(),
            direction_array.tolist(),
            parent_index=self.map.calc_index(current),
            cost=cost,
            steer=steer,
        )

        return node

    def check_car_collision(
        self,
        x_list: NDArray[np.float64],
        y_list: NDArray[np.float64],
        yaw_list: NDArray[np.float64],
    ) -> bool:
        # 路径分辨率小于网格分辨率，减少冗余计算
        sparse_idx = np.arange(0, len(x_list), 4)
        x_sparse = x_list[sparse_idx]
        y_sparse = y_list[sparse_idx]
        yaw_sparse = yaw_list[sparse_idx]

        cos_yaw = np.cos(yaw_sparse)
        sin_yaw = np.sin(yaw_sparse)
        bubble_centers = np.column_stack(
            (
                x_sparse + C.BUBBLE_DIST * cos_yaw,
                y_sparse + C.BUBBLE_DIST * sin_yaw,
            )
        )  # (N, 2)
        all_ids: List[List[int]] = self.obstacle_kd_tree.query_ball_point(
            bubble_centers,
            C.BUBBLE_R * A.SAFETY_MARGIN_RATIO,  # 在原始障碍物地图上，扩张机器人半径查询，基本等价于膨胀障碍物
            p=2,
            return_sorted=True,  # 让障碍点按距离排序，这样后面矩形判断有更大的概率提前退出
        )  # 每一个中心点对应一个List[int]
        if any(len(ids) > 0 for ids in all_ids):
            return False

        # 原实现中额外进行矩形检测，但是感觉性价比不高，
        # 使用圆形包络（kdtree直接查询）虽然有一些保守，但是感觉还可以接受。
        # valid_mask = [len(ids) > 0 for ids in all_ids]
        # valid_idx = np.flatnonzero(valid_mask)
        # for idx in valid_idx:
        #     idx: int
        #     obs_pts = self.map.kd_tree_points[all_ids[idx], :]
        #     if not self.rectangle_check(x_sparse[idx], y_sparse[idx], yaw_sparse[idx], obs_pts):
        #         return False
        return True

    def rectangle_check(self, x: float, y: float, yaw: float, ob_points: NDArray[np.float64]) -> bool:
        # 传进来的点数组为(N,2)，转换为齐次坐标(N,3)
        cos_y = cos(yaw)
        sin_y = sin(yaw)
        SE2inv_T = np.array(
            [
                [cos_y, sin_y, -cos_y * x - sin_y * y],
                [-sin_y, cos_y, sin_y * x - cos_y * y],
            ]
        ).T
        # SE2inv: 表示在本体坐标系，SE2最后一行用不到，省略
        # SE2inv_T: 直接生成转置的版本
        ob_points_homo = np.empty((ob_points.shape[0], 3), dtype=np.float64)
        ob_points_homo[:, :2] = ob_points
        ob_points_homo[:, 2] = 1.0

        # transformed = SE2inv @ ob_points_homo  # 正常应该是这个式子，但是会引入两次转置
        transformed = ob_points_homo @ SE2inv_T  # (N,2)
        for rx, ry in transformed:
            if not (rx > C.LF or rx < -C.LB or ry > C.W / 2.0 or ry < -C.W / 2.0):
                return False  # collision
        return True  # no collision

    def calc_heuristic(self, n: HNode, goal: HNode) -> float:
        # * atar启发式
        ind = self.map.calc_index_2d(n)
        h_star = (
            999999999 if ind not in self.hstar_dp else self.hstar_dp[ind].cost * A.XY_GRID_RESOLUTION
        )  # 这里有量纲变换
        h_rs = 0

        # * rs启发式：离线打表
        h_rs = 0.0  # 表里没有
        if self.hrs_table is not None:
            x = goal.x_list[0] - n.x_list[0]
            y = goal.y_list[0] - n.y_list[0]
            yaw = goal.yaw_list[0] - n.yaw_list[0]
            if x < 0:
                x = -x
                yaw = -yaw
            if y < 0:
                y = -y
                yaw = -yaw

            x_idx = round(x / A.XY_GRID_RESOLUTION)
            y_idx = round(y / A.XY_GRID_RESOLUTION)
            yaw_idx = round((yaw + pi) / A.YAW_GRID_RESOLUTION)

            # 越界检查
            x_size, y_size, yaw_size = self.hrs_table.shape
            if 0 <= x_idx < x_size and 0 <= y_idx < y_size and 0 <= yaw_idx < yaw_size:
                h_rs = self.hrs_table[x_idx, y_idx, yaw_idx]
                if math.isinf(h_rs):
                    h_rs = 999999

        # * rs启发式：另一种在线计算方案
        # path = rs.reeds_shepp_path_planning(
        #     n.x_list[0],
        #     n.y_list[0],
        #     n.yaw_list[0],
        #     goal.x_list[0],
        #     goal.y_list[0],
        #     goal.yaw_list[0],
        #     maxc=math.tan(C.MAX_STEER) / C.WB,
        #     step_size=A.MOTION_RESOLUTION,
        # )
        # h_rs = 999999999 if path is None else path.L

        # ? 对于障碍物较多的环境，貌似h_rs没有什么机会大于h_star，不过打表查询对性能影响很小
        # if h_rs > h_star:
        #     pass

        # *两者取最大
        return max(h_star, h_rs)

    @staticmethod
    def calc_rs_path_cost(reed_shepp_path: rs.RPath):
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

    @staticmethod
    def calc_rs_interval(h: float) -> int:
        """根据启发项 h 计算 RS 路径的扩展间隔 N(h)，分段线性下降策略。"""
        if h >= A.H_HIGH:
            return A.N_MAX
        elif h <= A.H_LOW:
            return A.N_MIN
        else:
            ratio = (h - A.H_LOW) / (A.H_HIGH - A.H_LOW)
            return math.ceil(A.N_MIN + (A.N_MAX - A.N_MIN) * ratio)

    @staticmethod
    def calc_cost(n: HNode, h: float) -> float:
        return n.cost + A.H_WEIGHT * h  # 启发项乘个权重

    def get_final_path(self, closed: Dict[int, HNode], goal_node: HNode) -> HPath:
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
        direction[0] = direction[1]  # adjust first direction

        path = HPath(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

        # 地图 -> 世界坐标转换
        out = np.stack([path.x_list, path.y_list, np.ones_like(path.x_list)], axis=0)  # shape: (3, N)
        global_xy = self.map.SE2 @ out  # map to world

        # 更新路径点为世界坐标
        path.x_list = global_xy[0].tolist()
        path.y_list = global_xy[1].tolist()

        # 角度变换（注意：地图内部角度是相对角度，要加回去）
        path.yaw_list = [((yaw - self.map.map_yaw + pi) % (2 * pi) - pi) for yaw in path.yaw_list]

        return path

    def smooth_path(self, path: HPath) -> HPath:
        #! 还没调好，可能不会用
        x = np.array(path.x_list)
        y = np.array(path.y_list)
        orig_x = np.copy(x)
        orig_y = np.copy(y)

        distance_map = self.map.edf_map
        H, W = distance_map.shape

        for _ in range(A.ITERATIONS):
            for i in range(1, len(x) - 1):  # skip endpoints
                # Smoothness term
                dx_s = x[i - 1] - 2 * x[i] + x[i + 1]
                dy_s = y[i - 1] - 2 * y[i] + y[i + 1]

                # Fidelity term
                dx_f = orig_x[i] - x[i]
                dy_f = orig_y[i] - y[i]

                # Obstacle term
                mx, my = self.map.world_to_map_2d(x[i], y[i])
                if 0 <= mx < W and 0 <= my < H:
                    d = distance_map[my, mx]
                    grad_obs = np.exp(-d / A.OBSTACLE_SIGMA) / (A.OBSTACLE_SIGMA + 1e-6)
                    # approximate gradient direction away from obstacle
                    gx = (distance_map[my, min(mx + 1, W - 1)] - distance_map[my, max(mx - 1, 0)]) / 2
                    gy = (distance_map[min(my + 1, H - 1), mx] - distance_map[max(my - 1, 0), mx]) / 2
                    dx_o = -grad_obs * gx
                    dy_o = -grad_obs * gy
                else:
                    dx_o = dy_o = 0.0

                # Total gradient
                dx = A.WEIGHT_SMOOTH * dx_s + A.WEIGHT_FIDELITY * dx_f + A.WEIGHT_OBSTACLE * dx_o
                dy = A.WEIGHT_SMOOTH * dy_s + A.WEIGHT_FIDELITY * dy_f + A.WEIGHT_OBSTACLE * dy_o

                x[i] += A.LEARN_RATE * dx
                y[i] += A.LEARN_RATE * dy

        # Recalculate yaw and direction
        new_yaw = []
        new_dir = []
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            new_yaw.append(atan2(dy, dx))
            new_dir.append(True)  # assume forward; refine if needed
        new_yaw.append(new_yaw[-1])
        new_dir.append(new_dir[-1])

        # Compute new cost (approximate length)
        cost = np.sum(np.hypot(np.diff(x), np.diff(y)))

        return HPath(x.tolist(), y.tolist(), new_yaw, new_dir, cost)


def main():
    print("Start Hybrid A* planning")

    # Set Initial parameters
    if not S.Debug.test_sim_origin:
        sim_origin = Pose2D(0, 0, 0)
        # start = Pose2D(40.0, 10.0, 90.0, deg=True)
        # goal = Pose2D(45.0, 35, 180.0, deg=True)
        start = Pose2D(-40.0, 10.0, 90.0, deg=True)
        goal = Pose2D(45.0, -35, 180.0, deg=True)
    else:
        # 这个测试接口处转换
        # TODO: 没测转角
        sim_origin = Pose2D(2, 5, 0, deg=True)
        start = Pose2D(42, 15, 90, deg=True)
        goal = Pose2D(47, 40, 180, deg=True)

    print("start : ", start)
    print("goal : ", goal)
    print("max curvature : ", tan(C.MAX_STEER) / C.WB)

    map = HMap(MAP_PASSABLE, origin=sim_origin)
    planner = HybridAStarPlanner(map)
    path, message = planner.planning(start, goal)
    if path is None:
        print(message)
        return

    x = [x / A.XY_GRID_RESOLUTION for x in path.x_list]
    y = [y / A.XY_GRID_RESOLUTION for y in path.y_list]
    if show_animation:
        fig, ax = plt.subplots()
        plot_path_curvature_map(path, ax)
        plot_binary_map(MAP_PASSABLE, ax)
        plt_tight_show()

    print(__file__ + " done!!")


def test_main():
    for _ in range(5):
        main()


if __name__ == "__main__":
    import datetime
    import sys
    from pathlib import Path as fPath

    import matplotlib.pyplot as plt
    from line_profiler import LineProfiler
    from plot.plot_utils import (
        MAP_PASSABLE,
        plot_binary_map,
        plot_path_curvature_map,
        plt_tight_show,
    )

    if S.Debug.use_profile:
        show_animation = False
        lp = LineProfiler()
        current_module = sys.modules[__name__]

        # lp.add_module(current_module)
        lp.add_function(HybridAStarPlanner.planning)
        # lp.add_function(HybridAStarPlanner.calc_cost)
        lp.add_function(HybridAStarPlanner.check_car_collision)
        # lp.add_function(HybridAStarPlanner.rectangle_check)
        lp.add_function(HybridAStarPlanner.update_node_with_analytic_expansion)
        lp.add_function(HybridAStarPlanner.calc_rs_path_cost)
        lp.add_function(HybridAStarPlanner.calc_motion_inputs)
        lp.add_function(HybridAStarPlanner.get_neighbors)
        lp.add_function(HybridAStarPlanner.calc_next_node)
        lp.add_function(HybridAStarPlanner.get_final_path)
        lp.add_function(rs.calc_paths)
        lp.add_function(calc_distance_heuristic)
        # lp.add_module(rs)

        lp_wrapper = lp(test_main)
        lp_wrapper()
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        name = fPath(__file__).stem
        short_name = "_".join(name.split("_")[:2])  # 取前两个单词组合
        profile_filename = f"profile_{short_name}_{timestamp}.txt"
        with open(profile_filename, "w", encoding="utf-8") as f:
            lp.print_stats(sort=True, stream=f)
    else:
        show_animation = True
        main()
