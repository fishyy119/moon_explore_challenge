"""

A* grid based planning

author: Nikos Kanargias (nkana@tee.gr)
and: me

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import heapq
import math
from typing import Dict, List, NamedTuple, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray


class ANode(NamedTuple):
    x_index: int
    y_index: int
    cost: float
    parent_index: int


class ANodeProto(Protocol):
    @property
    def x_index(self) -> int: ...
    @property
    def y_index(self) -> int: ...


class AMapProto(Protocol):
    min_x_index: int
    min_y_index: int
    x_width: int
    euclidean_dilated_ob_map: NDArray[np.bool_]
    resolution: float

    def calc_index_2d(self, node: ANodeProto) -> int: ...


def calc_distance_heuristic(map: AMapProto, gx: float, gy: float):
    """
    从终点反向搜索，提前构建出地图中所有栅格到终点的启发距离

    Args:
        map (Map): 地图
        gx (float): 终点坐标 [m]
        gy (float): 终点坐标 [m]
    """
    # dx, dy, cost
    sqr2 = math.sqrt(2)
    motions: List[Tuple[int, int, float]] = [
        (1, 0, 1),
        (0, 1, 1),
        (-1, 0, 1),
        (0, -1, 1),
        (-1, -1, sqr2),
        (-1, 1, sqr2),
        (1, -1, sqr2),
        (1, 1, sqr2),
    ]
    goal_node = ANode(
        round(gx / map.resolution),
        round(gy / map.resolution),
        0.0,
        -1,
    )
    open_set: Dict[int, ANode] = dict()
    closed_set: Dict[int, ANode] = dict()
    goal_idx = map.calc_index_2d(goal_node)
    open_set[goal_idx] = goal_node
    priority_queue: List[Tuple[float, int]] = [(0, goal_idx)]

    min_x = map.min_x_index
    min_y = map.min_y_index
    xw = map.x_width
    pq_push = heapq.heappush

    # 手动为四周加上障碍物（仅在此函数内部有效），优化掉循环中的越界判断
    # 因此终点需要在调用端提前进行越界判断
    # IDEA: 限制泛洪搜索在一个局部范围？
    ob_map = map.euclidean_dilated_ob_map.copy()
    ob_map[[0, -1], :] = True
    ob_map[:, [0, -1]] = True

    while True:
        if not priority_queue:
            break
        _, c_id = heapq.heappop(priority_queue)
        if c_id in open_set:
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue

        get_cost = current.cost
        get_x = current.x_index
        get_y = current.y_index
        for dx, dy, dc in motions:
            nx = get_x + dx
            ny = get_y + dy
            if ob_map[ny, nx]:  # 此处不需要越界判断，前面进行了预处理
                continue

            # 将 calc_index_2d 内联展开
            n_id = (ny - min_y) * xw + (nx - min_x)
            if n_id in closed_set:
                continue

            new_cost = get_cost + dc
            if (n_id not in open_set) or (open_set[n_id].cost >= new_cost):
                open_set[n_id] = ANode(nx, ny, new_cost, c_id)
                pq_push(priority_queue, (new_cost, n_id))

    return closed_set
