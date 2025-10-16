import pickle
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
from animated_models import AnalyticFrame, ExpandFrame, FinalFrame, Frame
from plot_utils import MAP_DEM, NPY_ROOT

from path_planner.hybrid_a_star_path import HPath
from path_planner.utils import A, C, Pose2D, S

# C = get_C()
C.MAX_PASSABLE_SLOPE = 15  # type: ignore
print(id(C))
import path_planner.rs_planning as rs
from path_planner.hybrid_a_star_map import HMap, HNode
from path_planner.hybrid_a_star_planner import HybridAStarPlanner as BasePlanner


class AnimatedPlanner(BasePlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames: List[Frame] = []

    def calc_heuristic(self, n: HNode, goal: HNode) -> float:
        h = super().calc_heuristic(n, goal)  # 这里将结果附加到了n.h_cost
        self.frames.append(ExpandFrame(n, self.calc_cost(n)))
        return h

    def get_final_path(self, closed: Dict[int, HNode], goal_node: HNode) -> HPath:
        path = super().get_final_path(closed, goal_node)
        self.frames.append(FinalFrame(path))

        return path

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
            # * =============================================================
            self.frames.append(AnalyticFrame(path, path_costs[idx]))
            # * =============================================================
            if self.check_car_collision(path.x, path.y, path.yaw):
                best_cost = path_costs[idx]
                best_path = path
                break

        # update
        if best_path is not None:
            # * =============================================================
            cast(AnalyticFrame, self.frames[-1]).success = True
            # * =============================================================
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


def main():
    print("Start Hybrid A* planning")

    start, goal = (Pose2D(27.0, 34.0, 180.0, deg=True), Pose2D(30.0, 43.0, 180.0, deg=True))

    print("start : ", start)
    print("goal : ", goal)
    print("max curvature : ", C.MAX_C)

    map = HMap(MAP_DEM, origin=Pose2D(0, 0, 0))
    planner = AnimatedPlanner(map)
    path, message = planner.planning(start, goal)
    if path is None:
        print(message)
        return

    print(__file__ + " done!!")
    with open(Path(__file__).parent / "frames.pkl", "wb") as f:
        pickle.dump(planner.frames, f)
    np.save(NPY_ROOT / "map_passable_new.npy", planner.map.obstacle_map)
    np.save(NPY_ROOT / "map_slope_new.npy", planner.map.slope_map)
    np.save(NPY_ROOT / "map_rought_new.npy", planner.map.rough_map)


if __name__ == "__main__":
    main()
