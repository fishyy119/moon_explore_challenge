import math

import rclpy
from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Path
from path_msgs.msg import HPath
from rclpy.node import Node
from std_msgs.msg import Header


class HPathVizNode(Node):
    def __init__(self):
        super().__init__("hpath_viz_node")
        self.sub = self.create_subscription(HPath, "/plan_path", self.hpath_callback, 10)  # 你发布HPath消息的topic名称
        self.pub = self.create_publisher(Path, "/plan_path_viz", 10)

    def hpath_callback(self, msg: HPath):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.poses = []

        for pose2d in msg.poses:
            pose2d: Pose2D
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = path_msg.header.frame_id
            pose_stamped.header.stamp = path_msg.header.stamp
            pose_stamped.pose.position.x = pose2d.x
            pose_stamped.pose.position.y = pose2d.y
            pose_stamped.pose.position.z = 0.0

            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = math.sin(pose2d.theta / 2)
            pose_stamped.pose.orientation.w = math.cos(pose2d.theta / 2)

            path_msg.poses.append(pose_stamped)

        self.pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HPathVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
