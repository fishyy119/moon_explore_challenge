cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0148 OLD)
project(path_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HPath.msg"
  "msg/CandidatePose2D.msg"
  "srv/PathPlanning.srv"
  "srv/ExplorePlanning.srv"
  DEPENDENCIES geometry_msgs
)

ament_package()
