import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from path_msg.msg import HPath
from path_planner.hybrid_a_star_planner import HMap, HybridAStarPlanner
from rclpy.node import Node
from utils import Pose2D as MyPose2D


class HybridAStarNode(Node):
    def __init__(self):
        super().__init__("_node")
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.path_pub = self.create_publisher(HPath, "/plan_path", 10)

        self.map = None
        self.planner = None

        self.get_logger().info("Hybrid A* Node initialized, waiting for map...")

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info("Received map")
        # 将ROS地图转为numpy bool数组（True为障碍物）
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        ob_map = data > 50  # 通常占用率大于50算作障碍物

        self.map = HMap(ob_map, resolution=resolution)
        self.planner = HybridAStarPlanner(self.map)

        # 示例起终点
        start = MyPose2D(1.0, 1.0, 0.0)
        goal = MyPose2D(8.0, 8.0, 0.0)
        path = self.planner.planning(start, goal)
        self.publish_path(path)

    def publish_path(self, path):
        msg = HPath()
        msg.directions = path.direction_list

        for x, y, yaw in zip(path.x_list, path.y_list, path.yaw_list):
            pose = Pose2D()
            pose.x = x
            pose.y = y
            pose.theta = yaw
            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.get_logger().info(f"Published path with {len(msg.poses)} poses")


def main(args=None):
    rclpy.init(args=args)
    node = HybridAStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
