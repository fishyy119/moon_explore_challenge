import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from path_msgs.msg import HPath as HPathROS
from path_msgs.srv import PathPlanning
from path_planner.hybrid_a_star_planner import HMap, HPath, HybridAStarPlanner
from rclpy.node import Node
from utils import Pose2D as MyPose2D


class HybridAStarNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")
        self.INFO = self.get_logger().info
        self.WARN = self.get_logger().warn
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        # self.path_pub = self.create_publisher(HPathROS, "/plan_path", 10)

        # 规划服务
        self.planning_srv = self.create_service(PathPlanning, "plan_path", self.handle_plan_path)

        self.map_msg = None
        self.map = None
        self.planner = None

        self.INFO("路径规划器初始化完毕")

    def handle_plan_path(self, request: PathPlanning.Request, response: PathPlanning.Response):
        if self.map_msg is None:
            response.success = False
            response.reason = "未收到地图"
            return response

        # 缓存的ROS地图信息，需要时再构造规划器实例
        msg = self.map_msg
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        ob_map = data > 50  # 通常占用率大于50算作障碍物

        self.map = HMap(ob_map, resolution=resolution)
        self.planner = HybridAStarPlanner(self.map)  # TODO: map_msg的origin

        start = MyPose2D(request.start.x, request.start.y, request.start.theta)
        goal = MyPose2D(request.goal.x, request.goal.y, request.goal.theta)

        path, plan_result = self.planner.planning(start, goal)
        if path is None:
            response.success = False
            response.reason = plan_result
            self.WARN(plan_result)
        else:
            response.success = True
            response.reason = f"规划路径成功"
            response.path = self.convert_to_hpath_msg(path)  # 路径放响应中

        return response

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg

    @staticmethod
    def convert_to_hpath_msg(path: HPath) -> HPathROS:
        msg = HPathROS()
        msg.directions = path.direction_list

        for x, y, yaw in zip(path.x_list, path.y_list, path.yaw_list):
            pose = Pose2D()
            pose.x = x
            pose.y = y
            pose.theta = yaw
            msg.poses.append(pose)

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = HybridAStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
