from dataclasses import dataclass

import path_planner.rs_planning as rs
from path_planner.hybrid_a_star_map import HNode
from path_planner.hybrid_a_star_path import HPath


@dataclass
class ExpandFrame:
    node: HNode
    total_cost: float


@dataclass
class AnalyticFrame:
    path: rs.RPath
    cost: float
    success: bool = False  # 是否通过碰撞检测


@dataclass
class FinalFrame:
    path: HPath


Frame = AnalyticFrame | ExpandFrame | FinalFrame
