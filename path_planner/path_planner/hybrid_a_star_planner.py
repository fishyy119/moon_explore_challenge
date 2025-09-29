"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)
and: me

"""

import heapq
import math
from math import cos, pi, sin, tan
from pathlib import Path as fPath
from typing import Dict, Generator, List, Set, Tuple, cast

import numpy as np
import rs_planning as rs
from dynamic_programming_heuristic import calc_distance_heuristic
from hybrid_a_star_map import HMap, HNode
from hybrid_a_star_path import HPath
from numpy.typing import NDArray
from utils import A, C, Pose2D, S


class HybridAStarPlanner:
    def __init__(self, map: HMap) -> None:
        self.map = map
        self.obstacle_kd_tree = map.build_kdtree()
        self.kd_tree_points = self.map.kd_tree_points
        self.hrs_table: NDArray[np.float64] | None

    def find_nearest_free_position(self, sx: float, sy: float) -> None | Tuple[float, float]:
        """
        若起点落入了障碍物（膨胀后）中，将其弹出到最近的自由栅格，偏航不变
        TODO: 未考虑内部有自由空间的环形障碍

        Args:
            sx (float): 起点坐标，单位为m
            sy (float): 起点坐标，单位为m

        Returns:
            Tuple[float,float] (Optional): 弹出后xy，同样为m
        """
        r_map = self.map.compute_point_distance_field(sx, sy)
        edf_map = self.map.edf_map
        free_mask = edf_map > self.map.rr * A.SAFETY_MARGIN_RATIO  # 这里要和HMap中的障碍物膨胀阈值保持一致
        masked_r_map = np.where(free_mask, r_map, np.inf)
        iy, ix = np.unravel_index(np.argmin(masked_r_map), masked_r_map.shape)

        # 拒绝弹出的过远的场景
        min_dist = masked_r_map[iy, ix] * self.map.resolution  # 换成米为单位，方便阈值判断
        if min_dist > A.MAX_POP_OUT_DISTANCE_RATIO * self.map.rr:
            return None

        return float(ix * self.map.resolution), float(iy * self.map.resolution)

    def planning(self, start: Pose2D, goal: Pose2D) -> Tuple[None | HPath, str]:
        """
        规划主函数
        输入都在外部地图坐标系表示，需要转换到内部地图坐标系（ROS地图消息的坐标系）
        !目前仅地图原点平移经过了测试，没有考虑存在旋转的情况
        Args:
            start (Pose2D): 起点位姿
            goal (Pose2D): 终点位姿
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
                f"输入坐标不在地图范围内：({start.x}, {start.y}) -> ({sidx[0]}, {sidx[1]}), ({goal.x}, {goal.y}) -> ({gidx[0]}, {gidx[1]})",
            )

        if self.map.euclidean_dilated_ob_map[sidx[1], sidx[0]]:
            new_xy = self.find_nearest_free_position(sx, sy)
            if new_xy is None:
                return None, "起点在障碍物中，且弹出失败"
            sx, sy = new_xy
            sidx = self.map.world_to_map(sx, sy, syaw)

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

        paths: List[rs.RPath] = rs.calc_paths(
            Pose2D(start_x, start_y, start_yaw),
            Pose2D(goal_x, goal_y, goal_yaw),
            C.MAX_C,
            step_size=self.map.resolution * A.MOTION_RESOLUTION_RATIO,
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
        MOTION_RESOLUTION = self.map.resolution * A.MOTION_RESOLUTION_RATIO
        arc_l = self.map.resolution * 1.5
        distance = MOTION_RESOLUTION * direction
        steps = int(np.ceil(arc_l / MOTION_RESOLUTION))
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

        # 坡度惩罚项
        node.slope_cost = tan(self.map.slope_map[node.y_index, node.x_index])
        node.rough_cost = self.map.rough_map[node.y_index, node.x_index]

        return node

    def check_car_collision(
        self,
        x_list: NDArray[np.floating],
        y_list: NDArray[np.floating],
        yaw_list: NDArray[np.floating],
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
            999999999 if ind not in self.hstar_dp else self.hstar_dp[ind].cost * self.map.resolution
        )  # 这里有量纲变换
        h_rs = 0

        # * rs启发式
        h_rs = rs.calc_rs_length(
            n.x_list[0],
            n.y_list[0],
            n.yaw_list[0],
            goal.x_list[0],
            goal.y_list[0],
            goal.yaw_list[0],
            maxc=C.MAX_C,
        )

        # ? 对于障碍物较多的环境，貌似h_rs没有什么机会大于h_star，不过此处不是性能瓶颈
        if h_rs > h_star:
            pass  # 调试时的一个记录断点

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
        return n.cost + A.SLOPE_WEIGHT * n.slope_cost + A.ROUGH_WEIGHT * n.rough_cost + A.H_WEIGHT * h  # 启发项乘个权重

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

    # def smooth_path(self, path: HPath) -> HPath:
    #     #! 还没调好，可能不会用
    #     x = np.array(path.x_list)
    #     y = np.array(path.y_list)
    #     orig_x = np.copy(x)
    #     orig_y = np.copy(y)

    #     distance_map = self.map.edf_map
    #     H, W = distance_map.shape

    #     for _ in range(A.ITERATIONS):
    #         for i in range(1, len(x) - 1):  # skip endpoints
    #             # Smoothness term
    #             dx_s = x[i - 1] - 2 * x[i] + x[i + 1]
    #             dy_s = y[i - 1] - 2 * y[i] + y[i + 1]

    #             # Fidelity term
    #             dx_f = orig_x[i] - x[i]
    #             dy_f = orig_y[i] - y[i]

    #             # Obstacle term
    #             mx, my = self.map.world_to_map_2d(x[i], y[i])
    #             if 0 <= mx < W and 0 <= my < H:
    #                 d = distance_map[my, mx]
    #                 grad_obs = np.exp(-d / A.OBSTACLE_SIGMA) / (A.OBSTACLE_SIGMA + 1e-6)
    #                 # approximate gradient direction away from obstacle
    #                 gx = (distance_map[my, min(mx + 1, W - 1)] - distance_map[my, max(mx - 1, 0)]) / 2
    #                 gy = (distance_map[min(my + 1, H - 1), mx] - distance_map[max(my - 1, 0), mx]) / 2
    #                 dx_o = -grad_obs * gx
    #                 dy_o = -grad_obs * gy
    #             else:
    #                 dx_o = dy_o = 0.0

    #             # Total gradient
    #             dx = A.WEIGHT_SMOOTH * dx_s + A.WEIGHT_FIDELITY * dx_f + A.WEIGHT_OBSTACLE * dx_o
    #             dy = A.WEIGHT_SMOOTH * dy_s + A.WEIGHT_FIDELITY * dy_f + A.WEIGHT_OBSTACLE * dy_o

    #             x[i] += A.LEARN_RATE * dx
    #             y[i] += A.LEARN_RATE * dy

    #     # Recalculate yaw and direction
    #     new_yaw = []
    #     new_dir = []
    #     for i in range(len(x) - 1):
    #         dx = x[i + 1] - x[i]
    #         dy = y[i + 1] - y[i]
    #         new_yaw.append(atan2(dy, dx))
    #         new_dir.append(True)  # assume forward; refine if needed
    #     new_yaw.append(new_yaw[-1])
    #     new_dir.append(new_dir[-1])

    #     # Compute new cost (approximate length)
    #     cost = np.sum(np.hypot(np.diff(x), np.diff(y)))

    #     return HPath(x.tolist(), y.tolist(), new_yaw, new_dir, cost)


def main():
    print("Start Hybrid A* planning")

    # Set Initial parameters
    if not S.Debug.test_sim_origin:
        sim_origin = Pose2D(0, 0, 0)
        start = Pose2D(40.0, 10.0, 90.0, deg=True)
        # start = Pose2D(44.0, 8.0, 90.0, deg=True)
        goal = Pose2D(45.0, 35.1, 180.0, deg=True)
        # start = Pose2D(-40.0, 10.0, 90.0, deg=True)
        # goal = Pose2D(45.0, -35, 180.0, deg=True)
    else:
        # 这个测试接口处转换
        sim_origin = Pose2D(2, 5, 0, deg=True)
        start = Pose2D(42, 15, 90, deg=True)
        goal = Pose2D(47, 40, 180, deg=True)

    print("start : ", start)
    print("goal : ", goal)
    print("max curvature : ", C.MAX_C)

    map = HMap(MAP_DEM, origin=sim_origin)
    planner = HybridAStarPlanner(map)
    path, message = planner.planning(start, goal)
    if path is None:
        print(message)
        return

    x = [x / map.resolution for x in path.x_list]
    y = [y / map.resolution for y in path.y_list]
    if show_animation:
        fig, ax = plt.subplots()
        plot_path_curvature_map(path, ax)
        plot_binary_map(map.obstacle_map, ax)
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
        MAP_DEM,
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
        lp.add_function(HybridAStarPlanner.calc_heuristic)
        lp.add_function(rs.calc_paths)
        lp.add_function(rs.calc_rs_length)
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
