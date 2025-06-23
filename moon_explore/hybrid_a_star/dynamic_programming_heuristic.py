"""

A* grid based planning

author: Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Protocol, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from utils import Settings


@dataclass(slots=True)
class ANode:
    x_index: int
    y_index: int
    cost: float
    parent_index: int


class ANodeProto(Protocol):
    x_index: int
    y_index: int


class AMapProto(Protocol):
    min_x_index: int
    max_x_index: int
    min_y_index: int
    max_y_index: int
    x_width: int
    euclidean_dilated_ob_map: NDArray[np.bool_]

    def world_to_map_2d(self, x: float, y: float) -> Tuple[int, int]: ...

    def calc_index_2d(self, node: ANodeProto) -> int: ...


def calc_distance_heuristic(map: AMapProto, gx: float, gy: float):
    """
    从终点反向搜索，提前构建出地图中所有栅格到终点的启发距离

    Args:
        map (Map): 地图
        gx (float): 终点坐标 [m]
        gy (float): 终点坐标 [m]
    """
    goal_node = ANode(*map.world_to_map_2d(gx, gy), 0.0, -1)
    motions = get_motion_model()

    open_set, closed_set = dict(), dict()
    open_set: Dict[int, ANode]
    closed_set: Dict[int, ANode]
    open_set[map.calc_index_2d(goal_node)] = goal_node
    priority_queue: List[Tuple[float, int]] = [(0, map.calc_index_2d(goal_node))]

    min_x = map.min_x_index
    max_x = map.max_x_index
    min_y = map.min_y_index
    max_y = map.max_y_index
    xw = map.x_width
    ob_map = map.euclidean_dilated_ob_map

    while True:
        if not priority_queue:
            break
        cost, c_id = heapq.heappop(priority_queue)
        if c_id in open_set:
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue

        for m in motions:
            nx = current.x_index + m[0]
            ny = current.y_index + m[1]
            new_cost = current.cost + m[2]

            if not (min_x <= nx < max_x and min_y <= ny < max_y):
                continue

            if ob_map[ny, nx]:
                continue

            # 将 calc_index_2d 内联展开
            n_id = (ny - min_y) * xw + (nx - min_x)

            if n_id in closed_set:
                continue

            new_nid: Dict[int, Tuple[int, int, float]] = {}
            if n_id not in open_set:
                new_nid[n_id] = (nx, ny, new_cost)
            else:
                if open_set[n_id].cost >= new_cost:
                    # This path is the best until now. record it!
                    new_nid[n_id] = (nx, ny, new_cost)

            for nid, args in new_nid.items():
                node = ANode(*args, c_id)
                open_set[nid] = node
                heapq.heappush(priority_queue, (node.cost, nid))

    return closed_set


def get_motion_model() -> List[Tuple[int, int, float]]:
    # dx, dy, cost
    motion: List[Tuple[int, int, float]] = [
        (1, 0, 1),
        (0, 1, 1),
        (-1, 0, 1),
        (0, -1, 1),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    return motion
