"""
Exploration target generator
"""

from dataclasses import dataclass
from itertools import zip_longest
from math import ceil, tan
from typing import List, Tuple

import numpy as np
import rs_planning as rs
from hybrid_a_star_planner import HMap
from plot.plot_utils import plot_canPoints_map
from scipy.stats import qmc
from utils import A, C, E, Pose2D


@dataclass
class CandidatePose:
    x: float
    y: float
    yaw: float = 0
    cost: float = 0


class ExplorePlanner:
    def __init__(self, map: HMap) -> None:
        self.map = map
        self.dilated_ob_map = map.euclidean_dilated_ob_map

    def planning(self, start: Pose2D, goal: Pose2D) -> List[Tuple[Pose2D, float]]:
        # * 1.采样区域计算
        base_radius = start - goal
        radius_list = [base_radius * r / self.map.resolution for r in E.R_RATIO_LIST]

        # 创建坐标网格并计算极坐标
        H, W = self.dilated_ob_map.shape
        x, y = np.meshgrid(np.arange(H), np.arange(W))
        dx = x - goal.x / self.map.resolution
        dy = y - goal.y / self.map.resolution
        r_map = np.sqrt(dx**2 + dy**2)

        # 生成每个圆环区域的 bool 掩码二维数组
        region_bounds = zip(radius_list[:-1], radius_list[1:])
        region_masks = [(r_map >= inner) & (r_map < outer) & (~self.dilated_ob_map) for inner, outer in region_bounds]
        region_areas = [
            np.count_nonzero(mask) * p
            for mask, p in zip_longest(region_masks, E.OUT_AREA_PANELTYS, fillvalue=min(E.OUT_AREA_PANELTYS))
        ]
        total_area = sum(region_areas)
        sample_nums = [ceil(E.SUM_SAMPLE_NUM * a / total_area) for a in region_areas]

        # * 2.采样
        total_required = sum(sample_nums)
        sampler = qmc.Halton(d=2, scramble=False)
        halton_points = sampler.random(n=total_required * 20)  # 多取一些以防止不落入区域内

        # 地图坐标范围（以像素计）
        sample_results: List[CandidatePose] = []
        counts = [0 for _ in sample_nums]

        for u, v in halton_points:
            # 归一化的采样局部窗口 映射到 地图像素坐标系
            # 貌似太靠近边缘采样效果不好，所以并不是严格的局部窗口，有1.5倍的扩张
            x_px = int(((u - 0.5) * base_radius * max(E.R_RATIO_LIST) * 1.5 + goal.x) / self.map.resolution)
            y_px = int(((v - 0.5) * base_radius * max(E.R_RATIO_LIST) * 1.5 + goal.y) / self.map.resolution)

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
                    theta = np.arctan2(goal.y - y, goal.x - x)
                    sample_results.append(CandidatePose(x, y, theta))
                    counts[region_idx] += 1
                    break  # 一个点只能算进一个区域

            if sum(counts) >= total_required:
                break
        # TODO:外面再加个循环保证能取够点？

        # * 3.计算代价
        for c in sample_results:
            paths_1: List[rs.RPath] = rs.calc_paths(
                start,
                Pose2D(c.x, c.y, c.yaw),
                tan(C.MAX_STEER) / C.WB,  # TODO: 加到配置项中
                step_size=A.MOTION_RESOLUTION,
            )
            paths_2: List[rs.RPath] = rs.calc_paths(
                Pose2D(c.x, c.y, c.yaw),
                goal,
                tan(C.MAX_STEER) / C.WB,  # TODO: 加到配置项中
                step_size=A.MOTION_RESOLUTION,
            )
            c.cost = max(p.L for p in paths_1) + max(p.L for p in paths_2) * 2

        return self.generate_result(sample_results)

    def generate_result(self, candidates: List[CandidatePose]) -> List[Tuple[Pose2D, float]]:
        result: List[Tuple[Pose2D, float]] = []
        for c in candidates:
            result.append((Pose2D(c.x, c.y, c.yaw), c.cost))
        return result


def main():
    start = Pose2D(42, 15, 90, deg=True)
    goal = Pose2D(47, 35, 180, deg=True)
    visiblemask = np.ones_like(MAP_PASSABLE, dtype=bool)
    H, W = visiblemask.shape
    for y in range(H):
        for x in range(W):
            if x + y > 800:  # 右上角三角形
                visiblemask[y, x] = False

    sim_origin = Pose2D(0, 0, 0)
    map = HMap(~(~MAP_PASSABLE & visiblemask), origin=sim_origin)
    planner = ExplorePlanner(map)
    results = planner.planning(start, goal)

    if True:
        fig, ax = plt.subplots()
        ax.scatter(start.x / A.XY_GRID_RESOLUTION, start.y / A.XY_GRID_RESOLUTION)
        ax.scatter(goal.x / A.XY_GRID_RESOLUTION, goal.y / A.XY_GRID_RESOLUTION, marker="*")
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
        plot_path_curvature_map,
        plt_tight_show,
    )

    main()
