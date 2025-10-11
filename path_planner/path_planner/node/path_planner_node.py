from typing import cast

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from grid_map_msgs.msg import GridMap
from numpy.typing import NDArray
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
from path_planner.utils import C
from path_planner.utils import Pose2D as MyPose2D
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

# from scipy.spatial.transform import Rotation as R


class HybridAStarNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")
        self.INFO = self.get_logger().info
        self.WARN = self.get_logger().warn
        self.ERR = self.get_logger().error
        self.map_sub = self.create_subscription(GridMap, "/elevation_map", self.map_callback, 10)
        self.path_pub = self.create_publisher(HPathROS, "/plan_path", 10)
        self.map_pub = self.create_publisher(GridMap, "/other_maps", 10)

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

    def generate_hmap(self) -> bool:
        if self.map_msg is None:
            self.INFO("未收到地图，无法进行规划")
            return False

        # 缓存的ROS地图信息，需要时再构造规划器实例
        msg = self.map_msg
        data_array: Float32MultiArray = msg.data[0]  # type: ignore

        # 获取行列信息
        resolution = msg.info.resolution
        length_x = msg.info.length_x
        length_y = msg.info.length_y
        n_cols = int(length_x / resolution)
        n_rows = int(length_y / resolution)

        # 将一维数据转二维
        flat_data = np.array(data_array.data, dtype=np.float32)
        if flat_data.size != n_cols * n_rows:
            self.WARN(f"高程图尺寸不符: {flat_data.size} != {n_cols} * {n_rows}")
            return False

        elevation = flat_data.reshape(n_rows, n_cols)

        # 计算原点偏移
        center_x = msg.info.pose.position.x
        center_y = msg.info.pose.position.y
        origin_x = center_x - length_x / 2.0
        origin_y = center_y - length_y / 2.0

        self.map = HMap(
            elevation,
            resolution=resolution,
            origin=MyPose2D(origin_x, origin_y, 0.0),
        )
        self.map_pub.publish(self.hmap_to_gridmap_msg(self.map, msg))  # 转发其内部计算地图
        return True

    def handle_plan_path(self, request: PathPlanning.Request, response: PathPlanning.Response):
        if self.map is None:
            if not self.generate_hmap():
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
            if not self.generate_hmap():
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

    def map_callback(self, msg: GridMap):
        self.map_msg = msg
        self.map = None

    @staticmethod
    def convert_to_hpath_msg(path: HPath) -> HPathROS:
        "将路径规划结果转换为 ROS2 消息"
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
    def hmap_to_gridmap_msg(hmap: HMap, elevation_msg: GridMap) -> GridMap:
        """将 HMap 处理后的高程图转换为 ROS2 GridMap 消息"""
        msg = GridMap()
        msg.header.frame_id = elevation_msg.header.frame_id
        msg.info = elevation_msg.info

        height, width = hmap.slope_map.shape  # 三个地图尺寸相同

        # === 添加三个图层 ===
        def make_layer(array: NDArray[np.floating]) -> Float32MultiArray:
            arr = Float32MultiArray()
            arr.layout.dim.append(MultiArrayDimension(label="rows", size=height, stride=height * width))  # type: ignore
            arr.layout.dim.append(MultiArrayDimension(label="cols", size=width, stride=width))  # type: ignore
            arr.data = array.astype(np.float32).flatten().tolist()
            return arr

        msg.layers = ["slope", "passable", "roughness"]
        msg.basic_layers = ["passable"]  # 可选：标记一个主要显示层
        msg.data = [
            make_layer(hmap.slope_map),
            make_layer(hmap.obstacle_map.astype(np.float32)),
            make_layer(hmap.rough_map),
        ]
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = HybridAStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
