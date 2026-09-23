# Pulled in by find_package(sightline_planner_cpp). The library's headers include
# MuJoCo's and Eigen's, and it links MuJoCo, so anything using it needs both found
# before the exported targets are read.
list(APPEND CMAKE_MODULE_PATH "${sightline_planner_cpp_DIR}")
find_package(Eigen3 REQUIRED)
find_package(MuJoCo REQUIRED)
