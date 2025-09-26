import math
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

ColorType = Union[Tuple[float, float, float, float], Tuple[float, float, float], str]
import sys
from typing import TYPE_CHECKING

sys.path.append(str(Path(__file__).parent.parent))

# * 在最一开始设置这个，保证后面的字体全部生效
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
# 感觉不生效
# plt.rcParams["mathtext.rm"] = "Times New Roman"  # 数学普通字体
# plt.rcParams["mathtext.it"] = "Times New Roman:italic"  # 数学斜体
# plt.rcParams["mathtext.bf"] = "Times New Roman:bold"  # 数学粗体
plt.rcParams.update({"axes.labelsize": 10.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})

try:
    NPY_ROOT = Path(__file__).parent.parent / "resource"
    MAP_PASSABLE = np.load(NPY_ROOT / "map_passable.npy")
    MAP_DEM = np.load(NPY_ROOT / "map_truth.npy").T
    MAP_EDF: NDArray[np.float64] = distance_transform_edt(~MAP_PASSABLE) / 10  # type: ignore
except:
    print("plot_utils未加载资源文件")
    pass


def ax_remove_axis(ax: Axes) -> None:
    # 对于绘制地图，去除坐标轴，添加黑色边框
    ax.axis("off")
    ax_add_black_border(ax, (0, 501), (0, 501))


def ax_set_square_lim(
    ax: Axes, xlim: Tuple[float, float] | None = None, ylim: Tuple[float, float] | None = None, border=False
):
    if xlim is None or ylim is None:
        ax.margins(x=0.05, y=0.05)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    x0, x1 = xlim
    y0, y1 = ylim
    w_x = x1 - x0
    w_y = y1 - y0
    if w_x > w_y:
        ylim = (y0 - (w_x - w_y) / 2, y1 + (w_x - w_y) / 2)
    else:
        xlim = (x0 - (w_y - w_x) / 2, x1 + (w_y - w_x) / 2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    if border:
        ax_add_black_border(ax, xlim, ylim)


def ax_add_black_border(ax: Axes, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    rect = patches.Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        linewidth=2,
        edgecolor="black",
        facecolor="none",
        transform=ax.transData,
    )
    ax.add_patch(rect)


def ax_add_legend(ax: Axes, legend_handles=None, alpha=1.0) -> None:
    # 自动设置图例样式
    # legend = ax.legend(handles=legend_handles, loc="upper right", title="")
    legend = ax.legend(
        handles=legend_handles,
        fontsize=10.5,
        prop={"family": ["SimSun", "Times New Roman"]},  # 中文宋体，西文 Times New Roman
        loc="best",
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(alpha)
    legend.get_frame().set_edgecolor("black")


def axes_add_abc(axes: List[Axes], y_offset=-0.05) -> None:
    # 添加图注 (a), (b)
    for i, ax in enumerate(axes):
        ax.text(
            0.5,
            y_offset,
            f"({chr(97 + i)})",
            transform=ax.transAxes,
            fontsize=10.5,  # 五号字体，用于图注
            fontname="Times New Roman",
            ha="center",
            va="top",
        )


def plt_tight_show(factor: float = 1) -> None:
    # A4 尺寸
    left_margin_mm = 30
    right_margin_mm = 26
    usable_width_cm = (210 - left_margin_mm - right_margin_mm) / 10  # mm → cm
    # 转为英寸
    fig_width_in = usable_width_cm / 2.54

    for i in plt.get_fignums():
        fig = plt.figure(i)
        fig.tight_layout()
        _, fig_height_in = fig.get_size_inches()
        fig.set_size_inches(fig_width_in * factor, fig_height_in)
        # 限制宽度，便于预览论文上的字体大小效果

        plt.tight_layout()

    plt.show()


def plt_flat_axes(axes: List[List[Axes]]) -> List[Axes]:
    # 将二维数组展平为一维列表
    flat_axes = [ax for axs in axes for ax in axs]
    return flat_axes


# def plot_slam_path_error(csv: RecordSLAM, ax: Axes):
#     x = csv.x.to_numpy()
#     y = csv.y.to_numpy()
#     errors = csv.xy_diff.to_numpy()

#     # 构造连续线段 [(x0,y0)-(x1,y1), (x1,y1)-(x2,y2), ...]
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)

#     # 创建 LineCollection，用距离值做颜色映射
#     lc = LineCollection(segments, cmap="viridis", norm=Normalize(vmin=min(errors), vmax=max(errors)))  # type: ignore
#     lc.set_array(np.array(errors))  # 将距离值传给颜色映射
#     lc.set_linewidth(2)
#     ax.add_collection(lc)

#     # 为了显示颜色条，创建一个辅助轴
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)

#     # 创建 colorbar 并设置样式
#     cb = ax.figure.colorbar(lc, cax=cax, orientation="vertical")  # type: ignore
#     cb.ax.set_title("误差 (m)", fontsize=10.5, pad=8)
#     # 设置 colorbar 的刻度为最小值、中间值和最大值
#     ticks = [min(errors), np.median(errors), max(errors)]
#     cb.set_ticks(ticks)
#     for label in cb.ax.get_yticklabels():
#         label.set_fontname("Times New Roman")
#     ax.axis("off")
#     ax_set_square_lim(ax, border=True)


# def plot_path_distance_map(csv: RecordBase, ax: Axes):
#     x = csv.x_map.to_numpy()
#     y = csv.y_map.to_numpy()
#     distances = [MAP_EDF[int(yi), int(xi)] for xi, yi in zip(x, y)]

#     # 构造连续线段 [(x0,y0)-(x1,y1), (x1,y1)-(x2,y2), ...]
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)

#     # 创建 LineCollection，用距离值做颜色映射
#     lc = LineCollection(segments, cmap="viridis", norm=Normalize(vmin=min(distances), vmax=max(distances)))  # type: ignore
#     lc.set_array(np.array(distances))  # 将距离值传给颜色映射
#     lc.set_linewidth(2)
#     ax.add_collection(lc)

#     # 为了显示颜色条，创建一个辅助轴
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)

#     # 创建 colorbar 并设置样式
#     cb = ax.figure.colorbar(lc, cax=cax, orientation="vertical")  # type: ignore
#     cb.ax.set_title("距离 (m)", fontsize=10.5, pad=8)
#     # 设置 colorbar 的刻度为最小值、中间值和最大值
#     ticks = [min(distances), np.median(distances), max(distances)]
#     cb.set_ticks(ticks)
#     for label in cb.ax.get_yticklabels():
#         label.set_fontname("Times New Roman")


if TYPE_CHECKING:
    from path_planner.hybrid_a_star_planner import HPath


def plot_path_curvature_map(path: "HPath", ax: Axes):
    x = np.array(path.x_list)
    y = np.array(path.y_list)
    yaw = np.array(path.yaw_list)

    dyaw = np.diff(np.unwrap(yaw))
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    ds[ds < 1e-5] = 1e6  # 防止除以0

    curvature = np.fabs(dyaw / ds)

    # 构造连续线段 [(x0,y0)-(x1,y1), (x1,y1)-(x2,y2), ...]
    points = np.array([x, y]).T.reshape(-1, 1, 2) * 10
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建 LineCollection，用时间做颜色映射
    lc = LineCollection(segments, cmap="viridis", norm=Normalize(curvature.min(), curvature.max()))  # type: ignore
    lc.set_array(curvature)
    lc.set_linewidth(2)
    ax.add_collection(lc)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # 创建 colorbar 并设置样式
    cb = ax.figure.colorbar(lc, cax=cax, orientation="vertical")  # type: ignore
    cb.ax.set_title("曲率", fontsize=10.5, pad=5)
    for label in cb.ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    # ====== 检测异常曲率点并绘制 ======
    mean_c = curvature.mean()
    std_c = curvature.std()
    threshold = mean_c + 3 * std_c

    abnormal_indices = np.where(curvature > threshold)[0]
    abnormal_x = (x[:-1] + x[1:]) / 2
    abnormal_y = (y[:-1] + y[1:]) / 2

    ax.scatter(
        abnormal_x[abnormal_indices] * 10,
        abnormal_y[abnormal_indices] * 10,
        color="red",
        s=20,
        marker="o",
    )


def plot_path_map(path: "HPath", ax: Axes):
    x = np.array(path.x_list)
    y = np.array(path.y_list)
    ax.plot(x * 10, y * 10)


def plot_binary_map(map: NDArray[np.bool_], ax: Axes, visible_map: NDArray[np.bool_] | None = None, alpha: float = 1):
    map_matrix = np.full_like(map, 0, dtype=int)  # 默认全部设为已知知区域 (0)
    map_matrix[map] = 1  # 障碍物区域 1
    if visible_map is not None:
        map_matrix[~visible_map] = -1

    # 自定义颜色映射，-1为灰色，0为白色，1为黑色
    cmap = mcolors.ListedColormap(["grey", "none", "black"])  # 只定义三种颜色
    bounds = [-1.5, -0.5, 0.5, 1.5]  # 设置每个值的边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 使用imshow绘制
    ax.imshow(map_matrix, cmap=cmap, norm=norm, alpha=alpha, origin="lower")
    ax_remove_axis(ax)


def plot_edf_map(map: NDArray[np.float64], ax: Axes, label_right: bool = True) -> None:
    im = ax.imshow(map, interpolation="nearest", origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = ax.figure.colorbar(im, cax=cax, orientation="vertical")  # type: ignore
    if label_right:  # 是把标签放到右边还是上边
        cb.ax.yaxis.set_label_text("距离 (m)")
    else:
        cb.ax.set_title("距离 (m)", fontsize=10.5, pad=8)
    for label in cb.ax.get_yticklabels():
        label.set_fontname("Times New Roman")
    ax_remove_axis(ax)


def plot_scenario_map(dem: NDArray[np.float64], ax: Axes) -> None:
    ax.imshow(dem, cmap="gist_yarg", origin="lower")
    ax_remove_axis(ax)


def plot_slope_map(slope: NDArray, ax: Axes, passable_threshold: List[float] = [5, 15, 20]) -> None:
    """
    绘制可通行性地图

    Args:
        slope (NDArray): 坡度地图，分四档
        passable_threshold (List[float], optional): 四档的分割点，与计算所用到的参数一致
    """
    slope_deg = np.degrees(np.arctan(slope))
    passability = np.ones_like(slope_deg).astype(np.int8)
    passability = np.digitize(slope_deg, passable_threshold, right=True).astype(np.int8)

    color_list = ["darkgreen", "lightgreen", "orange", "red"]
    cmap = plt.cm.colors.ListedColormap(color_list)  # type: ignore
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)  # type: ignore

    ax.imshow(passability, cmap=cmap, norm=norm, origin="lower")

    # 创建图例
    legend_labels = [
        f"0-{passable_threshold[0]}°",
        f"{passable_threshold[0]}-{passable_threshold[1]}°",
        f"{passable_threshold[1]}-{passable_threshold[2]}°",
        f">{passable_threshold[2]}°",
    ]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(color_list, legend_labels)]

    ax_add_legend(ax, legend_handles=legend_patches, alpha=0.8)
    ax_remove_axis(ax)


from path_planner.utils import Pose2D, Settings


def plot_pose2d_map(
    pose: Pose2D, ax: Axes, color: ColorType, scale: int = 20, resolution: float = Settings.A.XY_GRID_RESOLUTION
) -> None:
    x = pose.x / resolution
    y = pose.y / resolution
    yaw = pose.yaw_rad

    # 为了让线段变得不显眼
    dx = np.cos(yaw) * 0.01
    dy = np.sin(yaw) * 0.01
    # 绘制箭头
    ax.annotate(
        text="",
        xy=(x + dx, y + dy),  # 箭头指向方向
        xytext=(x, y),  # 箭头起点
        arrowprops=dict(
            arrowstyle="fancy",
            color=color,
            mutation_scale=scale,
            shrinkA=0,
            shrinkB=0,
            path_effects=[pe.withStroke(linewidth=2, foreground="gray")],  # 添加外框
        ),
        zorder=5,
    )


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if isinstance(x, list):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(
            x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width
        )
        plt.plot(x, y)


def plot_canPoints_map(
    canPoints: List[Tuple[Pose2D, float]], ax: Axes, resolution: float = Settings.A.XY_GRID_RESOLUTION, scale: int = 15
):
    # 将分数映射为颜色
    cmap = cm.get_cmap("viridis")
    scores = [s for p, s in canPoints]
    min_score, max_score = min(scores), max(scores)
    norm = Normalize(vmin=min_score, vmax=max_score)
    for point, score in canPoints:
        rgba_color = cmap(norm(score))
        plot_pose2d_map(point, ax=ax, color=rgba_color, scale=scale, resolution=resolution)

    # 创建伪图像对象以生成 colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # cb = ax.figure.colorbar(sm, cax=cax)  # type: ignore

    cb = ax.figure.colorbar(sm, cax=cax, orientation="vertical")  # type: ignore
    cb.ax.set_title("代价", fontsize=10.5, pad=5)
    # cb.ax.yaxis.set_tick_params(labelsize=10)
    for label in cb.ax.get_yticklabels():
        label.set_fontname("Times New Roman")
