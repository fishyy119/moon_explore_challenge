import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from geometry_msgs.msg import Point, Pose, Pose2D, Quaternion
from nav_msgs.msg import MapMetaData, OccupancyGrid
from path_msgs.srv import ExplorePlanning, PathPlanning
from path_planner.utils import A
from rclpy.node import Node
from rclpy.task import Future
from std_msgs.msg import Header


class TestPlannerNode(Node):
    def __init__(self):
        super().__init__("test_planner_node")

        # Step 1: 发布地图
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 10)
        self.map_msg = self.load_map_from_txt(
            Path(get_package_share_directory("path_planner")) / "resource/map_raw.txt"
        )
        self.map_pub.publish(self.map_msg)
        print(self.map_msg.info)
        self.get_logger().info("已发布地图")

        # Step 2: 创建客户端
        self.explore_cli = self.create_client(ExplorePlanning, "explore_planning")
        self.path_cli = self.create_client(PathPlanning, "path_planning")

        self.create_timer(1.0, self.run_test)  # 延迟1秒执行 run_test

    def load_map_from_txt(self, filepath: str | Path, resolution: float = 0.05) -> OccupancyGrid:
        data_2d = np.loadtxt(filepath, dtype=np.int8)

        height, width = data_2d.shape
        data_flat = data_2d.flatten().tolist()

        map_msg = OccupancyGrid()
        map_msg.header = Header()
        map_msg.header.frame_id = "map"

        map_msg.info.width = width
        map_msg.info.height = height
        map_msg.info.resolution = 0.01

        # origin 默认设置为 (0,0,0)
        map_msg.info.origin = Pose(
            position=Point(x=-2.673631, y=-3.71092987, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        )

        map_msg.data = data_flat
        return map_msg

    def run_test(self):
        self.get_logger().info("运行测试")

        # 检查 explore_planning 服务是否用可
        if not self.explore_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("找不到 explore_planning 服务")
            return

        explore_req = ExplorePlanning.Request()
        start = Pose2D(x=0.0, y=0.0, theta=0.0)
        explore_req.start = start
        explore_req.goal = Pose2D(x=6.0, y=6.0, theta=0.0)

        # 异步调用 explore
        explore_future = self.explore_cli.call_async(explore_req)
        explore_future.add_done_callback(lambda f: self._on_explore_done(f, start))

    def _on_explore_done(self, future: Future, start: Pose2D):
        try:
            res: ExplorePlanning.Response = future.result()
        except Exception as e:
            self.get_logger().error(f"explore_planning 服务调用出错: {e}")
            return

        self.get_logger().info(f"得到 {len(res.candidates)} 个候选点")
        if not res.candidates:
            self.get_logger().error("没有候选点")
            return

        candidate = res.candidates[0]

        if not self.path_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("找不到 path_planning 服务")
            return

        path_req = PathPlanning.Request()
        path_req.start = start
        path_req.goal = candidate.pose

        path_future = self.path_cli.call_async(path_req)
        path_future.add_done_callback(self._on_path_done)

    def _on_path_done(self, future: Future):
        try:
            res: PathPlanning.Response = future.result()
        except Exception as e:
            self.get_logger().error(f"path_planning 服务调用出错: {e}")
            return

        if not res.success:
            self.get_logger().error(f"path_planning 服务失败: {res.reason}")
            return

        self.get_logger().info(f"路径长度: {len(res.path.poses)}")


def main():
    rclpy.init()
    node = TestPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
