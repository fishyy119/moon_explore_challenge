from typing import List, Tuple

import numpy as np
from plot.plot_utils import *

from path_planner.hybrid_a_star_map import HMap
from path_planner.hybrid_a_star_planner import HybridAStarPlanner
from path_planner.utils import Pose2D


def generate_competition_hmap(indices: List[int]) -> HMap:
    """生成一个带有指定障碍方格的 5m×5m 测试 HMap"""
    size_m = 5.0
    res = 0.01
    # 初始化空地图
    n = int(size_m / res)
    ob_map = np.zeros((n, n), dtype=np.bool_)
    h, w = ob_map.shape
    half_h, half_w = h // 2, w // 2

    # 取右下角区域
    sub_h, sub_w = h - half_h, w - half_w
    cell_h, cell_w = sub_h // 3, sub_w // 3

    for idx in indices:
        if not (0 <= idx < 9):
            raise ValueError(f"Index {idx} out of range (0~8)")

        i, j = divmod(idx, 3)  # i 行, j 列
        y1 = i * cell_h
        y2 = (i + 1) * cell_h
        x1 = half_w + j * cell_w
        x2 = half_w + (j + 1) * cell_w

        ob_map[y1:y2, x1:x2] = True

    hmap = HMap(ob_map, resolution=res, region=(0, 0))
    return hmap


def visualize_competition_hmaps(all_indices: List[List[int]]) -> None:
    """
    根据传入的 9 组索引列表生成 9 个 HMap，
    并以 3×3 子图形式绘制。

    参数:
        all_indices: 包含 9 个元素的列表，每个元素是一个 indices 列表。
                     例如 [[0], [1, 2], [3, 4], ..., [8]]
    """
    if len(all_indices) != 9:
        raise ValueError("all_indices 必须包含 9 组索引")

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    fig.suptitle("Competition HMap Visualization", fontsize=14)

    for k, indices in enumerate(all_indices):
        i, j = divmod(k, 3)
        hmap = generate_competition_hmap(indices)
        planner = HybridAStarPlanner(hmap)
        path, _ = planner.planning(Pose2D(1.67, 0.87, 0), Pose2D(4.12, 3.89, 0))
        ax: Axes = axes[i, j]
        plot_binary_map(hmap.obstacle_map, ax)
        if path is not None:
            ax.plot([p / 0.01 for p in path.x_list], [p / 0.01 for p in path.y_list])
        else:
            print(indices)
        ax.axis("on")

    plt_tight_show()


if __name__ == "__main__":
    visualize_competition_hmaps(
        [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ]
    )
