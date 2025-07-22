# path_planner

> 最后编辑日期：2025-07-22

该包实现了基于混合 A* 算法的路径规划模块与探索规划模块，提供服务接口供其他模块请求路径，同时支持路径的可视化功能。

------

## 节点功能说明

###  `path_planner_node`

本节点为路径规划的核心节点，提供规划服务，内部集成了混合 A* 算法的调用逻辑。

#### 接口说明

| 接口类型 | 名称               | 消息/服务类型                   | 描述                                             |
| -------- | ------------------ | ------------------------------- | ------------------------------------------------ |
| 订阅     | `/map`             | `nav_msgs/OccupancyGrid`        | 建图节点提供的2D栅格地图                         |
| 发布     | `/plan_path`       | `path_msgs/msg/HPath`           | 规划成功后，规划结果会<u>转发</u>到该话题        |
| 服务     | `path_planning`    | `path_msgs/srv/PathPlanning`    | 路径规划的服务端，**路径规划需要通过该服务触发** |
| 服务     | `explore_planning` | `path_msgs/srv/ExplorePlanning` | 探索规划的服务端，在可通行地图中采样生成中间目标 |

###  `hpath_viz_node`

路径可视化辅助节点，用于将 `HPath` 类型的自定义路径消息转换为标准 `nav_msgs/Path` 消息，方便在 RViz 中进行可视化展示。

#### 接口说明

| 接口类型 | 名称             | 消息/服务类型         | 描述                             |
| -------- | ---------------- | --------------------- | -------------------------------- |
| 订阅     | `/plan_path`     | `path_msgs/msg/HPath` | 订阅路径规划节点发布的路径结果   |
| 发布     | `/plan_path_viz` | `nav_msgs/msg/Path`   | 转换后的标准格式，用于 RViz 展示 |

## 文件说明

- **ROS 节点文件**
  - `path_planner_node.py`：路径规划与探索规划的服务节点
  
  - `hpath_viz_node.py`：路径可视化辅助节点

  - `test_planner_node.py`：对两个服务的简单测试
  
- **核心功能模块**
  - `hybrid_a_star_planner.py`：路径规划算法的**主入口**

  - `exploration_planner.py`：探索规划的主入口

  - `utils.py`：辅助功能模块，包含**配置项类 `Settings`**，每项配置均附有注释说明

- **其他模块**
- `dynamic_programming_heuristic.py`：A*启发代价计算
  
- `rs_planning.py` / `rs_patterns.py`：Reeds-Shepp 路径计算
  
- `preprocess/rs_precompute.py`：打表预生成RS启发代价
  
- ...

