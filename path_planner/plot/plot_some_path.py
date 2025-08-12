from plot_utils import *

from path_planner.hybrid_a_star_planner import HMap, HybridAStarPlanner
from path_planner.utils import Pose2D

# 模拟不同的路径规划条件
scenarios = [
    # (start, goal)
    (Pose2D(40.0, 15.0, 90.0, deg=True), Pose2D(45.0, 32.0, 180.0, deg=True)),
    (Pose2D(40.0, 15.0, 180.0, deg=True), Pose2D(45.0, 32.0, 180.0, deg=True)),
    (Pose2D(40.0, 15.0, 270.0, deg=True), Pose2D(45.0, 32.0, 180.0, deg=True)),
    (Pose2D(40.0, 15.0, 0.0, deg=True), Pose2D(45.0, 32.0, 180.0, deg=True)),
    (Pose2D(30.0, 25.0, 90.0, deg=True), Pose2D(12.0, 10.0, 180.0, deg=True)),
    (Pose2D(30.0, 25.0, 180.0, deg=True), Pose2D(12.0, 10.0, 180.0, deg=True)),
    (Pose2D(30.0, 25.0, 270.0, deg=True), Pose2D(12.0, 10.0, 180.0, deg=True)),
    (Pose2D(30.0, 25.0, 0.0, deg=True), Pose2D(12.0, 10.0, 180.0, deg=True)),
]
# 创建地图和规划器
sim_origin = Pose2D(0, 0, 0)
map_data = HMap(MAP_PASSABLE, origin=sim_origin)
planner = HybridAStarPlanner(map_data)

rows, cols = 2, 4
fig_height_per_row = 2  # 每行高度
fig_width = cols * fig_height_per_row
fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height_per_row * rows))
plt.subplots_adjust(hspace=0.2)  # 调整行间距（越小越紧凑，默认大约 0.4）
axes = axes.ravel()  # 展平成一维列表，方便循环

# 绘制每个场景
for idx, (start, goal) in enumerate(scenarios):
    print(idx, end="")
    path, message = planner.planning(start, goal)
    ax = axes[idx]
    ax: Axes

    assert path is not None, message
    plot_path_map(path, ax)
    ax.margins(x=0.05, y=0.05)
    # 后面绘制地图会重置坐标轴范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plot_binary_map(MAP_PASSABLE, ax)
    ax_set_square_lim(ax, xlim, ylim, border=True)

# plt_tight_show()
plt.savefig("hybrid_astar_path.svg", bbox_inches="tight", dpi=600)
