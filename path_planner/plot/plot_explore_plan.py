from plot_utils import *

from path_planner.exploration_planner import ExplorePlanner
from path_planner.hybrid_a_star_planner import HMap
from path_planner.utils import E, Pose2D

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

base_radius = np.hypot(start.x - goal.x, start.y - goal.y)  # 起点到终点的欧氏距离
radius_list = [base_radius * r / map.resolution for r in E.R_RATIO_LIST]

fig, ax = plt.subplots(dpi=600)
for r in radius_list:
    circle = patches.Circle(
        (goal.x / map.resolution, goal.y / map.resolution),  # 圆心
        radius=r,  # 半径
        fill=False,  # 不填充
        color="red",  # 圆颜色
        linestyle="--",  # 虚线
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(circle)
ax.scatter(start.x / map.resolution, start.y / map.resolution, s=200)
ax.scatter(goal.x / map.resolution, goal.y / map.resolution, marker="*", s=200)
plot_binary_map(MAP_PASSABLE, ax, visiblemask)
plot_canPoints_map(results, ax, scale=20)
# ax_add_black_border(ax, (240, 500), (140, 400))
ax_set_square_lim(ax, (240, 500), (140, 400), border=True)
# plt_tight_show()
plt.savefig("explore_plan.svg", bbox_inches="tight", dpi=600)
