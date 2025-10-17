import ast
from typing import List, Tuple, cast

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from path_msgs.msg import CandidatePose2D
from path_msgs.msg import HPath as HPathROS
from path_msgs.srv import ExplorePlanning, GenerateTemplatePath, PathPlanning
from path_planner.exploration_planner import ExplorePlanner
from path_planner.hybrid_a_star_path import (
    HPath,
    generate_circle_path,
    generate_figure8_path,
    generate_forward_path,
)
from path_planner.hybrid_a_star_planner import HMap, HybridAStarPlanner
from path_planner.rs_planning import reeds_shepp_path_planning
from path_planner.utils import A, C
from path_planner.utils import Pose2D as MyPose2D
from rclpy.node import Node

# from scipy.spatial.transform import Rotation as R


class HybridAStarNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")
        self.INFO = self.get_logger().info
        self.WARN = self.get_logger().warn
        # 声明参数（初始默认值）
        self.declare_parameter("quick_obstacles", "")
        self.declare_parameter("precise_obstacles", "")

        self.quick_ob = self.parse_obstacles("quick_obstacles")
        self.precise_ob = self.parse_obstacles("precise_obstacles")

        self.INFO(f"Quick obstacles: {self.quick_ob}")
        self.INFO(f"Precise obstacles: {self.precise_ob}")

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.path_pub = self.create_publisher(HPathROS, "/plan_path", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/other_map", 10)

        # 规划服务
        self.planning_srv = self.create_service(PathPlanning, "path_planning", self.handle_plan_path)
        self.explore_srv = self.create_service(ExplorePlanning, "explore_planning", self.handle_explore_plan)
        self.template_srv = self.create_service(
            GenerateTemplatePath, "generate_template_path", self.handle_template_path
        )

        self.map_msg = None
        self.map = None
        self.path_planner = None

        self.INFO("路径规划器初始化完毕")

    def parse_obstacles(self, name: str) -> List[Tuple[float, float, float]]:
        """
        解析 CLI 传入的障碍物参数：
        - 一级用分号分隔
        - 二级用逗号分隔
        - 自动转换为 List[Tuple[float,float,float]]
        """
        value_str = self.get_parameter(name).get_parameter_value().string_value
        if not value_str:
            return []

        try:
            s = value_str.replace(";", "],[")
            s = f"[[{s}]]"
            parsed = ast.literal_eval(s)

            # 检查格式
            if isinstance(parsed, list) and all(isinstance(item, list) and len(item) == 3 for item in parsed):
                return [tuple(float(x) for x in item) for item in parsed]  # type: ignore
            else:
                self.get_logger().error(f"{name}: 每个子列表长度必须为3")
                return []
        except Exception as e:
            self.get_logger().error(f"参数解析失败 '{name}': {e}")
            return []

    def handle_template_path(self, request: GenerateTemplatePath.Request, response: GenerateTemplatePath.Response):
        # 将 geometry_msgs/Pose2D 转为你的自定义 Pose2D
        start_pose = MyPose2D(x=request.start_pose.x, y=request.start_pose.y, yaw=request.start_pose.theta)

        # 参数校验
        MIN_RADIUS = 1 / C.MAX_C  # 半径最小值

        # 根据 mode 调用对应模板函数
        if request.mode == "forward":
            hpath = generate_forward_path(start_pose, distance=request.distance, step=request.step)
        elif request.mode == "circle":
            radius = request.radius
            if radius <= MIN_RADIUS:
                self.WARN(f"半径{request.radius}过小，使用{MIN_RADIUS}")
                radius = MIN_RADIUS
            hpath = generate_circle_path(start_pose, radius=radius, arc_angle=request.arc_angle, step=request.step)
        elif request.mode == "figure8":
            hpath = generate_figure8_path(start_pose, radius=request.radius, step=request.step)
        else:
            self.WARN(f"未知模式 '{request.mode}', 使用默认值 forward")
            hpath = generate_forward_path(start_pose, distance=request.distance, step=request.step)

        # 将 HPath 转为 ROS2 消息
        path_msg = self.convert_to_hpath_msg(hpath)
        response.path = path_msg  # 路径放响应中
        self.path_pub.publish(path_msg)
        self.INFO("发布路径")

        return response

    def generate_hmap(self, abnormal: bool) -> bool:
        if self.map_msg is None:
            self.INFO("未收到地图，无法进行规划")
            return False

        # 缓存的ROS地图信息，需要时再构造规划器实例
        msg = self.map_msg
        info = msg.info
        width = info.width
        height = info.height
        resolution = info.resolution
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        if abnormal:
            ob_map = data > A.OCCUPANCY_THRESHOLD  # 仅障碍物
        else:
            ob_map = (data > A.OCCUPANCY_THRESHOLD) | (data == -1)  # 未知区域和障碍物

        origin = info.origin

        # * 不考虑yaw
        # q = origin.orientation
        # quat = [q.x, q.y, q.z, q.w]
        # r = R.from_quat(quat)
        # roll, pitch, yaw = r.as_euler("xyz")
        self.map = HMap(
            ob_map,
            resolution=resolution,
            origin=MyPose2D(origin.position.x, origin.position.y, 0),
            quick_ob=self.quick_ob,
            precise_ob=self.precise_ob,
        )
        self.map_pub.publish(self.hmap_to_gridmap_msg(self.map, msg))
        return True

    def handle_plan_path(self, request: PathPlanning.Request, response: PathPlanning.Response):
        if request.bad:
            res = self.map_msg.info.resolution if self.map_msg is not None else 0.01
            path = reeds_shepp_path_planning(
                request.start.x,
                request.start.y,
                request.start.theta,
                request.goal.x,
                request.goal.y,
                request.goal.theta,
                maxc=C.MAX_C,
                step_size=res * A.MOTION_RESOLUTION_RATIO,
            )
            if path is None:
                response.success = False
                response.reason = "规划失败"
                return response
            else:
                response.success = True
                response.reason = f"bad路径成功"
                path_msg = self.convert_to_hpath_msg(HPath.from_rpath(path))
                response.path = path_msg  # 路径放响应中
                self.path_pub.publish(path_msg)
                self.INFO("发布bad路径")
                return response

        if (self.map is None) or request.abnormal:
            if not self.generate_hmap(abnormal=request.abnormal):
                response.success = False
                response.reason = "未收到地图"
                return response
        self.map = cast(HMap, self.map)
        path_planner = HybridAStarPlanner(self.map)

        start = MyPose2D(request.start.x, request.start.y, request.start.theta)
        goal = MyPose2D(request.goal.x, request.goal.y, request.goal.theta)

        path, plan_result = path_planner.planning(start, goal)
        if path is None:
            response.success = False
            response.reason = plan_result
            self.WARN(plan_result)
        else:
            response.success = True
            response.reason = f"规划路径成功"
            path_msg = self.convert_to_hpath_msg(path)
            response.path = path_msg  # 路径放响应中
            self.path_pub.publish(path_msg)
            self.INFO("发布路径")

        return response

    def handle_explore_plan(self, request: ExplorePlanning.Request, response: ExplorePlanning.Response):
        if self.map is None:
            if not self.generate_hmap(False):
                return response
        self.map = cast(HMap, self.map)
        explore_planner = ExplorePlanner(self.map)

        start = MyPose2D(request.start.x, request.start.y, request.start.theta)
        goal = MyPose2D(request.goal.x, request.goal.y, request.goal.theta)

        candidates = explore_planner.planning(start, goal)
        for pose, score in candidates:
            ps_msg = CandidatePose2D()
            ps_msg.pose.x = pose.x
            ps_msg.pose.y = pose.y
            ps_msg.pose.theta = pose.yaw_rad
            ps_msg.score = score
            response.candidates.append(ps_msg)  # type: ignore

        self.INFO(f"探索规划完成，共生成 {len(response.candidates)} 个候选点")
        return response

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map = None
        self.generate_hmap(False)  # 调试用，即时响应地图订阅

    @staticmethod
    def convert_to_hpath_msg(path: HPath) -> HPathROS:
        msg = HPathROS()
        msg.directions = path.direction_list

        for x, y, yaw in zip(path.x_list, path.y_list, path.yaw_list):
            pose = Pose2D()
            pose.x = x
            pose.y = y
            pose.theta = yaw
            msg.poses.append(pose)  # type: ignore

        return msg

    @staticmethod
    def hmap_to_gridmap_msg(hmap: HMap, map_msg: OccupancyGrid) -> OccupancyGrid:
        """将 HMap 处理后的高程图转换为 ROS2 GridMap 消息"""
        msg = OccupancyGrid()
        msg.header = map_msg.header
        msg.info = map_msg.info

        ob_map = hmap.obstacle_map.astype(np.uint8)
        data = np.where(ob_map, 90, 0).astype(np.int8)
        msg.data = data.flatten().tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = HybridAStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
