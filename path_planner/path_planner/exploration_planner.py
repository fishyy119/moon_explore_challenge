"""
Exploration target generator
"""

from dataclasses import dataclass
from itertools import zip_longest
from math import ceil, pi
from typing import List, Tuple

import numpy as np
import rs_planning as rs
from hybrid_a_star_planner import HMap
from scipy.stats import qmc
from utils import A, C, E, Pose2D, S


@dataclass
class CandidatePose:
    x: float
    y: float
    yaw: float
    is_goal: bool = False  # 这个候选点是否就是可行终点
    failed: bool = False  # 代价计算是否失败
    cost: float = 0


class ExplorePlanner:
    def __init__(self, map: HMap) -> None:
        self.map = map
        self.dilated_ob_map = map.euclidean_dilated_ob_map

    def planning(self, start: Pose2D, goal: Pose2D) -> List[Tuple[Pose2D, float]]:
        # * 0.坐标转换
        out = np.array([[start.x, goal.x], [start.x, goal.y], [1, 1]])
        inn = self.map.SE2inv @ out
        sx = inn[0, 0]
        sy = inn[1, 0]
        syaw = (self.map.map_yaw + start.yaw_rad + pi) % (2 * pi) - pi
        gx = inn[0, 1]
        gy = inn[1, 1]
        gyaw = (self.map.map_yaw + goal.yaw_rad + pi) % (2 * pi) - pi

        # * 1.采样区域计算
        # *根据半径划分为若干环带区域，根据面积分配各区域采样点数，对外侧区域面积进行惩罚
        base_radius = start - goal
        radius_list = [base_radius * r / self.map.resolution for r in E.R_RATIO_LIST]

        # 生成每个圆环区域的 bool 掩码二维数组
        r_map = self.map.compute_point_distance_field(gx, gy)
        region_bounds = zip(radius_list[:-1], radius_list[1:])
        region_masks = [(r_map >= inner) & (r_map < outer) & (~self.dilated_ob_map) for inner, outer in region_bounds]
        region_areas = [
            np.count_nonzero(mask) * p
            for mask, p in zip_longest(region_masks, E.OUT_AREA_PANELTYS, fillvalue=min(E.OUT_AREA_PANELTYS))
        ]

        # * 2.采样
        # *利用低差异序列Halton采样，尽量保证均匀，该采样在0~1矩形中采样，映射到采样区域后根据可通行区域进行拒绝
        total_area = sum(region_areas)
        sample_results: List[CandidatePose] = []
        # 终点可达的情况下，终点也添加进去
        gx_px = round(gy / self.map.resolution)
        gy_px = round(gx / self.map.resolution)
        if (
            self.map.min_x_index <= gx_px < self.map.max_x_index
            and self.map.min_y_index <= gy_px < self.map.max_y_index
        ):
            if not self.dilated_ob_map[gy_px, gx_px]:
                sample_results.append(CandidatePose(gx, gy, gyaw, is_goal=True))

        if total_area > 0:
            sample_nums = [ceil(E.SUM_SAMPLE_NUM * a / total_area) for a in region_areas]

            total_required = sum(sample_nums)
            sampler = qmc.Halton(d=2, scramble=False)
            halton_points = sampler.random(n=total_required * 20)  # 多取一些以防止不落入区域内
            counts = [0 for _ in sample_nums]

            for u, v in halton_points:
                # 归一化的采样局部窗口 映射到 地图像素坐标系
                # 貌似太靠近边缘采样效果不好，所以并不是严格的局部窗口，有1.5倍的扩张
                x_px = int(((u - 0.5) * base_radius * max(E.R_RATIO_LIST) * 1.5 + gx) / self.map.resolution)
                y_px = int(((v - 0.5) * base_radius * max(E.R_RATIO_LIST) * 1.5 + gy) / self.map.resolution)

                if not (
                    self.map.min_x_index <= x_px < self.map.max_x_index
                    and self.map.min_y_index <= y_px < self.map.max_y_index
                ):
                    continue
                if self.dilated_ob_map[y_px, x_px]:
                    continue

                for region_idx, (mask, max_count) in enumerate(zip(region_masks, sample_nums)):
                    if counts[region_idx] >= max_count:
                        continue

                    if mask[y_px, x_px]:
                        # 记录：将像素坐标转换为世界坐标
                        x = x_px * self.map.resolution
                        y = y_px * self.map.resolution
                        theta = np.arctan2(gy - y, gx - x)
                        sample_results.append(CandidatePose(x, y, theta))
                        counts[region_idx] += 1
                        break  # 一个点只能算进一个区域

                if sum(counts) >= total_required:
                    break
        else:
            print("未找到自由空间用于采样")

        # * 3.计算代价
        # *计算不考虑障碍的rs路径长度作为代价，由已知段（起点到候选点）与未知段（候选点到终点）组成，对未知段长度惩罚
        for c in sample_results:
            try:
                paths_1: List[rs.RPath] = rs.calc_paths(
                    Pose2D(sx, sy, syaw),
                    Pose2D(c.x, c.y, c.yaw),
                    C.MAX_C,
                    step_size=self.map.resolution * A.MOTION_RESOLUTION_RATIO,
                )
                paths_2: List[rs.RPath] = rs.calc_paths(
                    Pose2D(c.x, c.y, c.yaw),
                    Pose2D(gx, gy, gyaw),
                    C.MAX_C,
                    step_size=self.map.resolution * A.MOTION_RESOLUTION_RATIO,
                )
                c.cost = min(p.L for p in paths_1) + min(p.L for p in paths_2) * E.UNKOWN_PATH_PANELTY
            except:
                c.failed = True

        return self.generate_result(sample_results)

    def generate_result(self, candidates: List[CandidatePose]) -> List[Tuple[Pose2D, float]]:
        result: List[Tuple[Pose2D, float]] = []
        for c in candidates:
            if not c.failed:
                global_xy = self.map.SE2 @ np.array([c.x, c.y, 1])
                global_yaw = (c.yaw - self.map.map_yaw + pi) % (2 * pi) - pi
                result.append((Pose2D(global_xy[0], global_xy[1], global_yaw), c.cost))
        result.sort(key=lambda x: x[1])
        return result


def main():
    start = Pose2D(42, 15, 90, deg=True)
    goal = Pose2D(47, 35, 180, deg=True)
    visiblemask = np.ones_like(MAP_PASSABLE, dtype=bool)
    H, W = visiblemask.shape
    for y in range(H):
        for x in range(W):
            if x + y > 750:  # 右上角三角形
                visiblemask[y, x] = False

    sim_origin = Pose2D(0, 0, 0, deg=True)
    map = HMap(~(~MAP_PASSABLE & visiblemask), origin=sim_origin)
    planner = ExplorePlanner(map)
    results = planner.planning(start, goal)

    if show_animation:
        fig, ax = plt.subplots()
        ax.scatter(start.x / map.resolution, start.y / map.resolution)
        ax.scatter(goal.x / map.resolution, goal.y / map.resolution, marker="*")
        plot_binary_map(MAP_PASSABLE, ax, visiblemask)
        plot_canPoints_map(results, ax)
        plt_tight_show()

    print(__file__ + " done!!")


if __name__ == "__main__":
    import datetime
    import sys
    from pathlib import Path as fPath

    import matplotlib.pyplot as plt
    from line_profiler import LineProfiler
    from plot.plot_utils import (
        MAP_PASSABLE,
        plot_binary_map,
        plot_canPoints_map,
        plot_path_curvature_map,
        plt_tight_show,
    )

    if S.Debug.use_profile:
        show_animation = False
        lp = LineProfiler()
        current_module = sys.modules[__name__]

        # lp.add_module(current_module)
        lp.add_function(ExplorePlanner.planning)

        lp_wrapper = lp(main)
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
