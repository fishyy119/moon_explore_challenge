"""
rs_precompute.py - Reeds-Shepp 启发式查找表预生成脚本
==============================================================

本文件用于预生成基于 Reeds-Shepp 曲线的启发式代价查找表，
该表格用于混合 A* 中的快速启发式估值。

表格维度为三维 (dx, dy, d_yaw)，其中：
    - dx ∈ [0, MAP_MAX_SIZE]
    - dy ∈ [0, MAP_MAX_SIZE]
    - d_yaw ∈ [−π, π]

其中利用到了两个对称性变换精简表格尺寸：
    - (x, y, yaw) -> (-x, y, -yaw)
    - (x, y, yaw) -> (x, -y, -yaw)

表格数据结构为：
    rs_table[x_index, y_index, yaw_index] = 从原点到该相对姿态的最短 RS 路径代价（无障碍）

生成后保存为：
    rs_table_<MAP_MAX_SIZE>x<MAP_MAX_SIZE>.npy

可视化函数 `plot_rs_table()` 支持对 yaw 维度投影后以图像方式展示最小、最大和平均启发式代价分布。

--------------------------------------------------------------
⚠️ 警告：若以下参数发生变化，为保证一致性需要重新运行 `precompute_rs_table()`
--------------------------------------------------------------

- A.MAP_MAX_SIZE           地图尺寸上限，影响 xy 方向表格范围
- A.XY_GRID_RESOLUTION     坐标分辨率，影响表格离散粒度（越小越精细）
- A.YAW_GRID_RESOLUTION    偏航角分辨率，决定 yaw 轴分片数
- A.MOTION_RESOLUTION      Reeds-Shepp 曲线分段精度，影响路径采样
- C.MAX_STEER              最大转向角（影响最小曲率半径）
- C.WB                     车辆轴距（wheelbase）

"""

import math
from pathlib import Path

import numpy as np
from rs_planning import reeds_shepp_path_planning
from utils import A, C, S

npy_file = Path(__file__).resolve().parent / f"rs_table_{A.MAP_MAX_SIZE}x{A.MAP_MAX_SIZE}.npy"


def precompute_rs_table():
    map_size = A.MAP_MAX_SIZE
    xy_res = A.XY_GRID_RESOLUTION
    yaw_res = A.YAW_GRID_RESOLUTION

    x_list = np.arange(0.0, map_size + 1e-6, xy_res)
    y_list = np.arange(0.0, map_size + 1e-6, xy_res)
    yaw_list = np.arange(-math.pi, math.pi - 1e-6, yaw_res)

    x_size = len(x_list)
    y_size = len(y_list)
    yaw_size = len(yaw_list)

    print(f"预期大小：{8 * x_size * y_size * yaw_size / 1024 / 1024:.2f}MB")
    print(f"维度: ({x_size}, {y_size}, {yaw_size})")

    rs_table = np.full((x_size, y_size, yaw_size), np.inf, dtype=np.float64)

    for xi, dx in enumerate(x_list):
        for yi, dy in enumerate(y_list):
            for yaw_i, dyaw in enumerate(yaw_list):
                path = reeds_shepp_path_planning(
                    sx=0,
                    sy=0,
                    syaw=0,
                    gx=dx,
                    gy=dy,
                    gyaw=dyaw,
                    maxc=math.tan(C.MAX_STEER) / C.WB,
                    step_size=A.MOTION_RESOLUTION,
                )
                cost = path.L if path is not None else np.inf
                rs_table[xi, yi, yaw_i] = cost

        print(f"{xi+1}/{x_size}", end="\r")

    np.save(npy_file, rs_table)


def plot_rs_table():
    import matplotlib.pyplot as plt

    # 加载表格
    table = np.load(npy_file)  # shape: (x, y, yaw)

    # 对 yaw 维度计算三种投影（对代价取最小、最大、平均）
    min_proj = np.min(table, axis=2).T  # (y, x)
    max_proj = np.max(table, axis=2).T
    mean_proj = np.mean(table, axis=2).T

    # 画图
    fig, axs = plt.subplots(1, 3)

    # 共用参数
    im_params = dict(origin="lower", interpolation="nearest", cmap="viridis")

    im0 = axs[0].imshow(min_proj, **im_params)
    axs[0].set_title("RS Heuristic Min over Yaw")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(max_proj, **im_params)
    axs[1].set_title("RS Heuristic Max over Yaw")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(mean_proj, **im_params)
    axs[2].set_title("RS Heuristic Mean over Yaw")
    fig.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    precompute_rs_table()
    # plot_rs_table()
