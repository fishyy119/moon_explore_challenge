cmake_minimum_required(VERSION 3.5)
project(path_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HPath.msg"
  "srv/PathPlanning.srv"
  DEPENDENCIES geometry_msgs
)

ament_package()
