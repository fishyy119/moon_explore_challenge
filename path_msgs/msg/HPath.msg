# HPath.msg

# 在路径上以一定步长采样得到的路径点序列，每个点描述了车辆后轴中心的xy坐标与车辆自身的偏航角
geometry_msgs/Pose2D[] poses 

# 相邻路径段的运动方向（`true` 为前进，`false` 为后退） 
bool[] directions